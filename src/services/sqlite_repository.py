"""
SQLite Credential Repository Service.

High-assurance, crash-resilient persistent storage for serialized encrypted
biometric credentials, revocation lists (CRL), and anti-replay nonce tracking.
Uses embedded zero-dependency SQLite (Python standard library `sqlite3`).
"""

import sqlite3
import threading
import time
from typing import Dict, Any, Optional, List

from src.services.types import (
    ProtectedCredentialDTO,
    SerializedCredentialEnvelope,
    BiometricErrorCode,
    BiometricServiceException,
    RevokedCredentialError
)
from src.services.credential_serializer import CredentialSerializer
from src.revocation.registry import TemplateRegistry


class SQLiteCredentialRepository:
    """
    ACID-compliant SQLite repository managing encrypted credential persistence,
    revocation status, and anti-replay nonce verification across service restarts.
    """
    def __init__(self, 
                 db_path: str = "credentials.db", 
                 serializer: Optional[CredentialSerializer] = None, 
                 registry: Optional[TemplateRegistry] = None):
        self.db_path = db_path
        self.serializer = serializer or CredentialSerializer()
        self.registry = registry or TemplateRegistry()
        self._lock = threading.RLock()
        
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        """Initializes database schema with WAL journal mode for high concurrency."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode = WAL;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS credentials (
                    subject_id TEXT PRIMARY KEY,
                    credential_id TEXT UNIQUE NOT NULL,
                    app_id TEXT NOT NULL,
                    num_triangles INTEGER NOT NULL,
                    raw_bytes BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    is_active INTEGER DEFAULT 1
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS revocations (
                    subject_id TEXT PRIMARY KEY,
                    credential_id TEXT NOT NULL,
                    revoked_at REAL NOT NULL,
                    reason TEXT NOT NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS seen_nonces (
                    nonce_tag TEXT PRIMARY KEY,
                    observed_at REAL NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_cred_cid ON credentials (credential_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nonces_time ON seen_nonces (observed_at);")
            self._conn.commit()

    def store(self, subject_id: str, envelope: SerializedCredentialEnvelope, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Stores an encrypted credential envelope, clearing any prior active credential for the subject."""
        cid = envelope.credential_id
        app_id = metadata.get("app_id", "default_app") if metadata else "default_app"
        num_triangles = metadata.get("num_triangles", 0) if metadata else 0
        now = time.time()

        with self._lock:
            cur = self._conn.cursor()
            # If subject was revoked, new enrollment (Option A re-issuance) reactivates them
            cur.execute("DELETE FROM revocations WHERE subject_id = ?;", (subject_id,))
            cur.execute("""
                INSERT OR REPLACE INTO credentials 
                (subject_id, credential_id, app_id, num_triangles, raw_bytes, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, 1);
            """, (subject_id, cid, app_id, num_triangles, envelope.raw_bytes, now))
            self._conn.commit()

            # Synchronize with in-memory TemplateRegistry if present
            self.registry.enroll_template(
                subject_id=subject_id,
                app_id=app_id,
                template={"credential_id": cid}
            )
            return cid

    def get_raw_bytes(self, subject_id: str) -> bytes:
        """Retrieves raw encrypted bytes if active; raises error if revoked or missing."""
        with self._lock:
            if self.is_revoked(subject_id):
                raise RevokedCredentialError(
                    BiometricErrorCode.CREDENTIAL_REVOKED,
                    f"Subject '{subject_id}' credential has been revoked and cannot be retrieved"
                )

            cur = self._conn.cursor()
            cur.execute("SELECT raw_bytes FROM credentials WHERE subject_id = ? AND is_active = 1;", (subject_id,))
            row = cur.fetchone()
            if not row:
                raise BiometricServiceException(
                    BiometricErrorCode.CREDENTIAL_NOT_FOUND,
                    f"No active credential found for subject '{subject_id}'"
                )
            return row[0]

    def get_credential(self, subject_id: str, storage_key: bytes, app_id: str = "") -> ProtectedCredentialDTO:
        """Retrieves and decrypts the active protected credential for a subject."""
        raw_bytes = self.get_raw_bytes(subject_id)
        return self.serializer.deserialize_protected(
            raw_bytes=raw_bytes,
            storage_key=storage_key,
            subject_id=subject_id,
            app_id=app_id
        )

    def revoke(self, subject_id: str, reason: str = "security_compromise") -> bool:
        """Revokes the active credential for a subject and records in persistent CRL."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT credential_id FROM credentials WHERE subject_id = ? AND is_active = 1;", (subject_id,))
            row = cur.fetchone()
            if not row:
                return False

            cid = row[0]
            now = time.time()
            cur.execute("UPDATE credentials SET is_active = 0 WHERE subject_id = ?;", (subject_id,))
            cur.execute("""
                INSERT OR REPLACE INTO revocations (subject_id, credential_id, revoked_at, reason)
                VALUES (?, ?, ?, ?);
            """, (subject_id, cid, now, reason))
            self._conn.commit()

            # Revoke in TemplateRegistry
            for tmpl_id, rec in self.registry._database.items():
                if rec.get("subject_id") == subject_id and rec.get("status") == "ACTIVE":
                    self.registry.revoke_template(tmpl_id, reason=reason)

            return True

    def is_revoked(self, subject_id: str) -> bool:
        """Returns True if subject exists in revocations table."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT 1 FROM revocations WHERE subject_id = ?;", (subject_id,))
            return cur.fetchone() is not None

    def count(self) -> int:
        """Returns number of active stored credentials."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM credentials WHERE is_active = 1;")
            return int(cur.fetchone()[0])

    def record_seen_nonce(self, nonce_tag: str) -> bool:
        """
        Records a seen nonce tag in the database for persistent replay prevention.
        Returns True if fresh, False if already seen (replay detected).
        """
        with self._lock:
            cur = self._conn.cursor()
            try:
                cur.execute("INSERT INTO seen_nonces (nonce_tag, observed_at) VALUES (?, ?);", (nonce_tag, time.time()))
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def is_nonce_seen(self, nonce_tag: str) -> bool:
        """Checks if a nonce tag has already been recorded."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT 1 FROM seen_nonces WHERE nonce_tag = ?;", (nonce_tag,))
            return cur.fetchone() is not None

    def close(self):
        """Closes the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

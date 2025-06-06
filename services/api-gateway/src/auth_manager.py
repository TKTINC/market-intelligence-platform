"""
Authentication and Authorization Manager
"""

import asyncio
import aioredis
import jwt
import bcrypt
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import secrets
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class User:
    user_id: str
    email: str
    username: str
    role: str
    tier: str
    permissions: List[str]
    created_at: datetime
    last_login: datetime
    is_active: bool
    metadata: Dict[str, Any]

@dataclass
class APIKey:
    key_id: str
    user_id: str
    key_name: str
    key_hash: str
    permissions: List[str]
    rate_limit_tier: str
    expires_at: Optional[datetime]
    created_at: datetime
    last_used: Optional[datetime]
    is_active: bool

class AuthManager:
    def __init__(self):
        self.redis = None
        
        # JWT configuration
        self.jwt_config = {
            "secret_key": "your_jwt_secret_key_here",  # Should be from environment
            "algorithm": "HS256",
            "access_token_expire_minutes": 60,
            "refresh_token_expire_days": 30
        }
        
        # User roles and permissions
        self.roles = {
            "user": {
                "permissions": [
                    "portfolio:read", "portfolio:create", "portfolio:update",
                    "trading:execute", "trading:history",
                    "analysis:request", "market_data:read"
                ],
                "tier": "basic"
            },
            "premium": {
                "permissions": [
                    "portfolio:read", "portfolio:create", "portfolio:update", "portfolio:delete",
                    "trading:execute", "trading:history", "trading:advanced",
                    "analysis:request", "analysis:advanced", "market_data:read", "market_data:realtime",
                    "risk:monitor", "risk:alerts"
                ],
                "tier": "premium"
            },
            "admin": {
                "permissions": [
                    "*"  # All permissions
                ],
                "tier": "enterprise"
            }
        }
        
        # API key tiers
        self.api_key_tiers = {
            "free": {"rate_limit": 100, "burst_limit": 10},
            "basic": {"rate_limit": 1000, "burst_limit": 50},
            "premium": {"rate_limit": 10000, "burst_limit": 200},
            "enterprise": {"rate_limit": 100000, "burst_limit": 1000}
        }
        
        # Session management
        self.active_sessions = {}
        self.blacklisted_tokens = set()
        
        # Security settings
        self.security_config = {
            "password_min_length": 8,
            "password_require_special": True,
            "max_login_attempts": 5,
            "lockout_duration_minutes": 30,
            "session_timeout_minutes": 120,
            "api_key_length": 32
        }
        
        # Authentication statistics
        self.auth_stats = {
            "total_logins": 0,
            "failed_logins": 0,
            "active_sessions": 0,
            "api_key_calls": 0,
            "blocked_attempts": 0
        }
        
    async def initialize(self):
        """Initialize the authentication manager"""
        try:
            # Initialize Redis connection
            self.redis = aioredis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )
            
            # Load active sessions
            await self._load_active_sessions()
            
            # Load blacklisted tokens
            await self._load_blacklisted_tokens()
            
            # Start background tasks
            asyncio.create_task(self._session_cleanup_task())
            asyncio.create_task(self._security_monitoring_task())
            
            logger.info("Authentication manager initialized")
            
        except Exception as e:
            logger.error(f"Authentication manager initialization failed: {e}")
            raise
    
    async def close(self):
        """Close the authentication manager"""
        if self.redis:
            await self.redis.close()
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                raise Exception("Token has been revoked")
            
            # Decode JWT token
            payload = jwt.decode(
                token,
                self.jwt_config["secret_key"],
                algorithms=[self.jwt_config["algorithm"]]
            )
            
            # Check token expiration
            exp = payload.get("exp")
            if exp and datetime.utcfromtimestamp(exp) < datetime.utcnow():
                raise Exception("Token has expired")
            
            # Get user data
            user_id = payload.get("user_id")
            if not user_id:
                raise Exception("Invalid token payload")
            
            user_data = await self._get_user_data(user_id)
            if not user_data:
                raise Exception("User not found")
            
            if not user_data.get("is_active", False):
                raise Exception("User account is disabled")
            
            # Update last seen
            await self._update_user_last_seen(user_id)
            
            return {
                "user_id": user_id,
                "email": user_data.get("email"),
                "username": user_data.get("username"),
                "role": user_data.get("role", "user"),
                "tier": user_data.get("tier", "basic"),
                "permissions": user_data.get("permissions", [])
            }
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            raise Exception("Invalid token")
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            raise
    
    async def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key and return user info"""
        
        try:
            # Hash the provided key for lookup
            key_hash = self._hash_api_key(api_key)
            
            # Look up API key data
            api_key_data = await self._get_api_key_data(key_hash)
            if not api_key_data:
                raise Exception("Invalid API key")
            
            # Check if API key is active
            if not api_key_data.get("is_active", False):
                raise Exception("API key is disabled")
            
            # Check expiration
            expires_at = api_key_data.get("expires_at")
            if expires_at and datetime.fromisoformat(expires_at) < datetime.utcnow():
                raise Exception("API key has expired")
            
            # Get user data
            user_id = api_key_data.get("user_id")
            user_data = await self._get_user_data(user_id)
            if not user_data:
                raise Exception("User not found")
            
            # Update API key last used
            await self._update_api_key_last_used(api_key_data["key_id"])
            
            # Update stats
            self.auth_stats["api_key_calls"] += 1
            
            return {
                "user_id": user_id,
                "email": user_data.get("email"),
                "username": user_data.get("username"),
                "role": user_data.get("role", "user"),
                "tier": api_key_data.get("rate_limit_tier", "basic"),
                "permissions": api_key_data.get("permissions", []),
                "auth_method": "api_key",
                "key_id": api_key_data["key_id"]
            }
            
        except Exception as e:
            logger.error(f"API key verification failed: {e}")
            raise
    
    async def login(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate user and create session"""
        
        try:
            # Check for too many failed attempts
            if await self._is_account_locked(email):
                raise Exception("Account temporarily locked due to too many failed attempts")
            
            # Get user by email
            user_data = await self._get_user_by_email(email)
            if not user_data:
                await self._record_failed_login(email)
                raise Exception("Invalid credentials")
            
            # Verify password
            if not self._verify_password(password, user_data.get("password_hash", "")):
                await self._record_failed_login(email)
                raise Exception("Invalid credentials")
            
            # Check if user is active
            if not user_data.get("is_active", False):
                raise Exception("User account is disabled")
            
            # Clear failed login attempts
            await self._clear_failed_login_attempts(email)
            
            # Create tokens
            access_token = await self._create_access_token(user_data)
            refresh_token = await self._create_refresh_token(user_data)
            
            # Create session
            session_id = await self._create_session(user_data["user_id"], access_token)
            
            # Update user last login
            await self._update_user_last_login(user_data["user_id"])
            
            # Update stats
            self.auth_stats["total_logins"] += 1
            self.auth_stats["active_sessions"] = len(self.active_sessions)
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": self.jwt_config["access_token_expire_minutes"] * 60,
                "session_id": session_id,
                "user": {
                    "user_id": user_data["user_id"],
                    "email": user_data["email"],
                    "username": user_data["username"],
                    "role": user_data["role"],
                    "tier": user_data["tier"]
                }
            }
            
        except Exception as e:
            logger.error(f"Login failed for {email}: {e}")
            raise
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        
        try:
            # Verify refresh token
            payload = jwt.decode(
                refresh_token,
                self.jwt_config["secret_key"],
                algorithms=[self.jwt_config["algorithm"]]
            )
            
            if payload.get("type") != "refresh":
                raise Exception("Invalid token type")
            
            user_id = payload.get("user_id")
            if not user_id:
                raise Exception("Invalid token payload")
            
            # Get user data
            user_data = await self._get_user_data(user_id)
            if not user_data or not user_data.get("is_active", False):
                raise Exception("User not found or disabled")
            
            # Create new access token
            access_token = await self._create_access_token(user_data)
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": self.jwt_config["access_token_expire_minutes"] * 60
            }
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid refresh token: {e}")
            raise Exception("Invalid refresh token")
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise
    
    async def logout(self, token: str) -> bool:
        """Logout user and invalidate token"""
        
        try:
            # Decode token to get user info
            payload = jwt.decode(
                token,
                self.jwt_config["secret_key"],
                algorithms=[self.jwt_config["algorithm"]]
            )
            
            user_id = payload.get("user_id")
            
            # Add token to blacklist
            self.blacklisted_tokens.add(token)
            await self.redis.sadd("blacklisted_tokens", token)
            
            # Remove active session
            session_id = None
            for sid, session in list(self.active_sessions.items()):
                if session.get("user_id") == user_id:
                    session_id = sid
                    break
            
            if session_id:
                await self._remove_session(session_id)
            
            # Update stats
            self.auth_stats["active_sessions"] = len(self.active_sessions)
            
            return True
            
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    async def create_api_key(
        self,
        user_id: str,
        key_name: str,
        permissions: List[str],
        rate_limit_tier: str = "basic",
        expires_in_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create new API key for user"""
        
        try:
            # Generate API key
            api_key = secrets.token_urlsafe(self.security_config["api_key_length"])
            key_hash = self._hash_api_key(api_key)
            
            # Set expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            
            # Create API key record
            key_id = f"key_{secrets.token_hex(8)}"
            api_key_data = APIKey(
                key_id=key_id,
                user_id=user_id,
                key_name=key_name,
                key_hash=key_hash,
                permissions=permissions,
                rate_limit_tier=rate_limit_tier,
                expires_at=expires_at,
                created_at=datetime.utcnow(),
                last_used=None,
                is_active=True
            )
            
            # Store API key
            await self._store_api_key(api_key_data)
            
            return {
                "key_id": key_id,
                "api_key": api_key,  # Only returned once
                "key_name": key_name,
                "permissions": permissions,
                "rate_limit_tier": rate_limit_tier,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "created_at": api_key_data.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"API key creation failed: {e}")
            raise
    
    async def revoke_api_key(self, user_id: str, key_id: str) -> bool:
        """Revoke an API key"""
        
        try:
            # Get API key data
            api_key_data = await self._get_api_key_by_id(key_id)
            if not api_key_data:
                return False
            
            # Check ownership
            if api_key_data.get("user_id") != user_id:
                return False
            
            # Mark as inactive
            api_key_data["is_active"] = False
            await self._store_api_key_data(key_id, api_key_data)
            
            return True
            
        except Exception as e:
            logger.error(f"API key revocation failed: {e}")
            return False
    
    async def check_permission(self, user_data: Dict[str, Any], required_permission: str) -> bool:
        """Check if user has required permission"""
        
        try:
            user_permissions = user_data.get("permissions", [])
            user_role = user_data.get("role", "user")
            
            # Admin has all permissions
            if user_role == "admin" or "*" in user_permissions:
                return True
            
            # Check specific permission
            if required_permission in user_permissions:
                return True
            
            # Check wildcard permissions
            permission_parts = required_permission.split(":")
            if len(permission_parts) == 2:
                wildcard_permission = f"{permission_parts[0]}:*"
                if wildcard_permission in user_permissions:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    async def get_user_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all API keys for a user"""
        
        try:
            api_key_keys = await self.redis.keys(f"api_key:*")
            user_api_keys = []
            
            for key in api_key_keys:
                api_key_data = await self.redis.get(key)
                if api_key_data:
                    data = json.loads(api_key_data)
                    if data.get("user_id") == user_id:
                        # Don't include the actual key hash
                        safe_data = {k: v for k, v in data.items() if k != "key_hash"}
                        user_api_keys.append(safe_data)
            
            return user_api_keys
            
        except Exception as e:
            logger.error(f"User API keys retrieval failed: {e}")
            return []
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        import hashlib
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def _create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        
        expire = datetime.utcnow() + timedelta(minutes=self.jwt_config["access_token_expire_minutes"])
        
        payload = {
            "user_id": user_data["user_id"],
            "email": user_data["email"],
            "role": user_data["role"],
            "type": "access",
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_config["secret_key"], algorithm=self.jwt_config["algorithm"])
    
    async def _create_refresh_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        
        expire = datetime.utcnow() + timedelta(days=self.jwt_config["refresh_token_expire_days"])
        
        payload = {
            "user_id": user_data["user_id"],
            "type": "refresh",
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_config["secret_key"], algorithm=self.jwt_config["algorithm"])
    
    async def _create_session(self, user_id: str, access_token: str) -> str:
        """Create user session"""
        
        session_id = f"session_{secrets.token_hex(16)}"
        
        session_data = {
            "user_id": user_id,
            "access_token": access_token,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        }
        
        self.active_sessions[session_id] = session_data
        
        # Store in Redis
        await self.redis.setex(
            f"session:{session_id}",
            self.security_config["session_timeout_minutes"] * 60,
            json.dumps(session_data)
        )
        
        return session_id
    
    async def _remove_session(self, session_id: str):
        """Remove user session"""
        
        self.active_sessions.pop(session_id, None)
        await self.redis.delete(f"session:{session_id}")
    
    async def _get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data from storage"""
        
        try:
            user_data = await self.redis.get(f"user:{user_id}")
            if user_data:
                return json.loads(user_data)
            return None
            
        except Exception as e:
            logger.error(f"User data retrieval failed: {e}")
            return None
    
    async def _get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user data by email"""
        
        try:
            # Get user ID from email mapping
            user_id = await self.redis.get(f"email_to_user:{email}")
            if user_id:
                return await self._get_user_data(user_id)
            return None
            
        except Exception as e:
            logger.error(f"User by email retrieval failed: {e}")
            return None
    
    async def _get_api_key_data(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """Get API key data by hash"""
        
        try:
            api_key_data = await self.redis.get(f"api_key_hash:{key_hash}")
            if api_key_data:
                return json.loads(api_key_data)
            return None
            
        except Exception as e:
            logger.error(f"API key data retrieval failed: {e}")
            return None
    
    async def _get_api_key_by_id(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get API key data by ID"""
        
        try:
            api_key_data = await self.redis.get(f"api_key:{key_id}")
            if api_key_data:
                return json.loads(api_key_data)
            return None
            
        except Exception as e:
            logger.error(f"API key by ID retrieval failed: {e}")
            return None
    
    async def _store_api_key(self, api_key_data: APIKey):
        """Store API key data"""
        
        try:
            data_dict = asdict(api_key_data)
            
            # Store by key ID
            await self.redis.set(
                f"api_key:{api_key_data.key_id}",
                json.dumps(data_dict, default=str)
            )
            
            # Store by hash for lookup
            await self.redis.set(
                f"api_key_hash:{api_key_data.key_hash}",
                json.dumps(data_dict, default=str)
            )
            
        except Exception as e:
            logger.error(f"API key storage failed: {e}")
            raise
    
    async def _store_api_key_data(self, key_id: str, api_key_data: Dict[str, Any]):
        """Store API key data by ID"""
        
        try:
            await self.redis.set(
                f"api_key:{key_id}",
                json.dumps(api_key_data, default=str)
            )
            
            # Also update hash lookup if key_hash exists
            if "key_hash" in api_key_data:
                await self.redis.set(
                    f"api_key_hash:{api_key_data['key_hash']}",
                    json.dumps(api_key_data, default=str)
                )
            
        except Exception as e:
            logger.error(f"API key data storage failed: {e}")
    
    async def _update_api_key_last_used(self, key_id: str):
        """Update API key last used timestamp"""
        
        try:
            api_key_data = await self._get_api_key_by_id(key_id)
            if api_key_data:
                api_key_data["last_used"] = datetime.utcnow().isoformat()
                await self._store_api_key_data(key_id, api_key_data)
            
        except Exception as e:
            logger.error(f"API key last used update failed: {e}")
    
    async def _update_user_last_seen(self, user_id: str):
        """Update user last seen timestamp"""
        
        try:
            await self.redis.set(f"user_last_seen:{user_id}", datetime.utcnow().isoformat())
            
        except Exception as e:
            logger.error(f"User last seen update failed: {e}")
    
    async def _update_user_last_login(self, user_id: str):
        """Update user last login timestamp"""
        
        try:
            user_data = await self._get_user_data(user_id)
            if user_data:
                user_data["last_login"] = datetime.utcnow().isoformat()
                await self.redis.set(f"user:{user_id}", json.dumps(user_data, default=str))
            
        except Exception as e:
            logger.error(f"User last login update failed: {e}")
    
    async def _is_account_locked(self, email: str) -> bool:
        """Check if account is locked due to failed attempts"""
        
        try:
            failed_attempts = await self.redis.get(f"failed_login:{email}")
            if failed_attempts:
                attempts_data = json.loads(failed_attempts)
                attempt_count = attempts_data.get("count", 0)
                last_attempt = datetime.fromisoformat(attempts_data.get("last_attempt", "2000-01-01"))
                
                # Check if lockout period has expired
                lockout_duration = timedelta(minutes=self.security_config["lockout_duration_minutes"])
                if datetime.utcnow() - last_attempt < lockout_duration:
                    return attempt_count >= self.security_config["max_login_attempts"]
            
            return False
            
        except Exception as e:
            logger.error(f"Account lock check failed: {e}")
            return False
    
    async def _record_failed_login(self, email: str):
        """Record failed login attempt"""
        
        try:
            current_data = await self.redis.get(f"failed_login:{email}")
            
            if current_data:
                attempts_data = json.loads(current_data)
                attempts_data["count"] += 1
            else:
                attempts_data = {"count": 1}
            
            attempts_data["last_attempt"] = datetime.utcnow().isoformat()
            
            await self.redis.setex(
                f"failed_login:{email}",
                self.security_config["lockout_duration_minutes"] * 60,
                json.dumps(attempts_data)
            )
            
            # Update stats
            self.auth_stats["failed_logins"] += 1
            
            if attempts_data["count"] >= self.security_config["max_login_attempts"]:
                self.auth_stats["blocked_attempts"] += 1
            
        except Exception as e:
            logger.error(f"Failed login recording failed: {e}")
    
    async def _clear_failed_login_attempts(self, email: str):
        """Clear failed login attempts for email"""
        
        try:
            await self.redis.delete(f"failed_login:{email}")
            
        except Exception as e:
            logger.error(f"Failed login clearing failed: {e}")
    
    async def _load_active_sessions(self):
        """Load active sessions from Redis"""
        
        try:
            session_keys = await self.redis.keys("session:*")
            
            for key in session_keys:
                session_data = await self.redis.get(key)
                if session_data:
                    session_id = key.split(":", 1)[1]
                    self.active_sessions[session_id] = json.loads(session_data)
            
            logger.info(f"Loaded {len(self.active_sessions)} active sessions")
            
        except Exception as e:
            logger.error(f"Active sessions loading failed: {e}")
    
    async def _load_blacklisted_tokens(self):
        """Load blacklisted tokens from Redis"""
        
        try:
            blacklisted_tokens = await self.redis.smembers("blacklisted_tokens")
            self.blacklisted_tokens.update(blacklisted_tokens)
            
            logger.info(f"Loaded {len(self.blacklisted_tokens)} blacklisted tokens")
            
        except Exception as e:
            logger.error(f"Blacklisted tokens loading failed: {e}")
    
    # Background tasks
    async def _session_cleanup_task(self):
        """Background task to cleanup expired sessions"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = datetime.utcnow()
                expired_sessions = []
                
                for session_id, session_data in list(self.active_sessions.items()):
                    last_activity = datetime.fromisoformat(session_data.get("last_activity", "2000-01-01"))
                    timeout_duration = timedelta(minutes=self.security_config["session_timeout_minutes"])
                    
                    if current_time - last_activity > timeout_duration:
                        expired_sessions.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    await self._remove_session(session_id)
                
                # Update stats
                self.auth_stats["active_sessions"] = len(self.active_sessions)
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
            except Exception as e:
                logger.error(f"Session cleanup task error: {e}")
                await asyncio.sleep(300)
    
    async def _security_monitoring_task(self):
        """Background task for security monitoring"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Monitor for suspicious activity
                current_time = datetime.utcnow()
                
                # Check for excessive failed login attempts
                failed_login_keys = await self.redis.keys("failed_login:*")
                
                high_risk_accounts = []
                for key in failed_login_keys:
                    attempts_data = await self.redis.get(key)
                    if attempts_data:
                        data = json.loads(attempts_data)
                        if data.get("count", 0) >= self.security_config["max_login_attempts"]:
                            email = key.split(":", 1)[1]
                            high_risk_accounts.append(email)
                
                if high_risk_accounts:
                    logger.warning(f"High risk accounts with failed logins: {len(high_risk_accounts)}")
                
                # Cleanup old blacklisted tokens
                # In a real implementation, we'd check JWT expiration times
                
            except Exception as e:
                logger.error(f"Security monitoring task error: {e}")
                await asyncio.sleep(3600)

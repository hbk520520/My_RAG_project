import json
import redis
from typing import Dict, Any, Optional

class StateManager:
    """Redis 状态管理器：脱水 (save) / 复水 (load)"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", expire_seconds: int = 3600):
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.expire = expire_seconds

    def save_state(self, session_id: str, state: Dict[str, Any]) -> None:
        """脱水：将内存状态写入 Redis"""
        self.client.setex(session_id, self.expire, json.dumps(state, ensure_ascii=False))

    def load_state(self, session_id: str) -> Dict[str, Any]:
        """复水：从 Redis 读取最新状态"""
        raw = self.client.get(session_id)
        if not raw:
            raise KeyError(f"Session {session_id} not found or expired")
        return json.loads(raw)
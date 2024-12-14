from redis_helper import RedisHelper
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_redis_connection():
    """Test Redis setup"""
    redis_helper = RedisHelper()
    
    # Test connection
    if not redis_helper.is_healthy():
        logger.error("Redis connection failed")
        return False

    # Test basic operations
    test_key = "test:key"
    test_value = {"test": "data"}
    
    # Test set
    if not redis_helper.cache_set(test_key, test_value):
        logger.error("Failed to set test data")
        return False

    # Test get
    retrieved = redis_helper.cache_get(test_key)
    if retrieved != test_value:
        logger.error("Retrieved data doesn't match")
        return False

    # Test delete
    if not redis_helper.cache_delete(test_key):
        logger.error("Failed to delete test data")
        return False

    # Get stats
    stats = redis_helper.get_stats()
    logger.info(f"Redis Stats: {stats}")

    logger.info("All Redis tests passed successfully!")
    return True

if __name__ == "__main__":
    test_redis_connection()
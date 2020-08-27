import redis
import json
db = redis.StrictRedis(host='0.0.0.0',
	port=6378, db=0)
while True:
    queue = db.lrange('music_dev', 0, 0)
    db.ltrim('music_dev', 1, -1)
    for q in queue:
        q = json.loads(q.decode("utf-8"))
        print(q)
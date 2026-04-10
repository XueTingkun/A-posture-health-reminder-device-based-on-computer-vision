import json
import os

DEFAULT_CONFIG = {
    "PITCH_THRESHOLD": -10,
    "ROLL_THRESHOLD": 25,
}


def _load_config():
    if not os.path.exists("./config/config.json"):
        os.makedirs("./config", exist_ok=True)
        with open("./config/config.json", "w+", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=True, separators=(",", ":"))
    with open("./config/config.json") as f:
        try:
            return json.load(f)
        except json.decoder.JSONDecodeError:
            return {}


_c = _load_config()

PITCH_THRESHOLD = _c.get("PITCH_THRESHOLD", DEFAULT_CONFIG["PITCH_THRESHOLD"])
ROLL_THRESHOLD = _c.get("ROLL_THRESHOLD", DEFAULT_CONFIG["ROLL_THRESHOLD"])

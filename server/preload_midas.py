from model_depth import get_midas_bundle


if __name__ == "__main__":
    bundle = get_midas_bundle()
    print(
        {
            "status": "ok",
            "model": "MiDaS_small",
            "device": str(bundle["device"]),
        }
    )

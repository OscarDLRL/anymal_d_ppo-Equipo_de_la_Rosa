from pathlib import Path
import argparse
import re

BASE_DIR = Path("pretrained_models/anymal_d")
VIDEOS_DIR = BASE_DIR / "videos"

reward_re = re.compile(r"Reward-([-+]?\d+(?:\.\d+)?)_policy\.pt$")


def get_reward(policy_path: Path):
    match = reward_re.search(policy_path.name)
    if not match:
        return None
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Borra de verdad. Sin esto solo muestra qué borraría.")
    parser.add_argument("--delete-videos", action="store_true", help="También borra videos e imágenes generadas.")
    args = parser.parse_args()

    if not BASE_DIR.exists():
        print(f"No existe la carpeta: {BASE_DIR}")
        return

    policies = list(BASE_DIR.glob("*_policy.pt"))

    if not policies:
        print("No encontré checkpoints *_policy.pt")
        return

    valid_policies = []
    for p in policies:
        reward = get_reward(p)
        if reward is not None:
            valid_policies.append((reward, p))

    if not valid_policies:
        print("Encontré policies, pero ninguna tiene Reward-xxx en el nombre.")
        print("No borraré nada por seguridad.")
        return

    best_reward, best_policy = max(valid_policies, key=lambda x: x[0])
    best_prefix = best_policy.name.replace("_policy.pt", "")
    best_optimizer = BASE_DIR / f"{best_prefix}_optimizer.pt"

    keep = {best_policy}
    if best_optimizer.exists():
        keep.add(best_optimizer)

    to_delete = []

    # Borra todos los .pt menos el mejor policy y su optimizer
    for f in BASE_DIR.glob("*.pt"):
        if f not in keep:
            to_delete.append(f)

    # Opcional: borra videos e imágenes generadas
    if args.delete_videos and VIDEOS_DIR.exists():
        for pattern in ["*.mp4", "*.png", "*.jpg", "*.jpeg"]:
            to_delete.extend(VIDEOS_DIR.glob(pattern))

    print("\nMejor checkpoint detectado:")
    print(f"  Policy:    {best_policy}")
    print(f"  Reward:    {best_reward}")
    if best_optimizer.exists():
        print(f"  Optimizer: {best_optimizer}")
    else:
        print("  Optimizer: no encontrado")

    print("\nArchivos que se conservarán:")
    for f in sorted(keep):
        print(f"  KEEP {f}")

    print("\nArchivos que se borrarían:")
    if not to_delete:
        print("  Nada que borrar.")
    else:
        total_mb = 0
        for f in sorted(to_delete):
            size_mb = f.stat().st_size / (1024 * 1024)
            total_mb += size_mb
            print(f"  DELETE {f}  ({size_mb:.2f} MB)")
        print(f"\nEspacio aproximado a liberar: {total_mb:.2f} MB")

    if not args.yes:
        print("\nModo seguro: no borré nada.")
        print("Para borrar de verdad corre:")
        print("  python clean_checkpoints.py --yes")
        print("O también borrar videos:")
        print("  python clean_checkpoints.py --yes --delete-videos")
        return

    for f in to_delete:
        f.unlink()

    print("\nListo. Se borraron los archivos seleccionados.")


if __name__ == "__main__":
    main()

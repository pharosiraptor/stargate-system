import os
import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python stargate.py <command>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "create":
        create_stargate()
    elif command == "destroy":
        destroy_stargate()
    else:
        print("Unknown command.")


def create_stargate():
    print("Creating Stargate...")
    # Placeholder for creating a Stargate
    subprocess.run(["echo", "Stargate created."])


def destroy_stargate():
    print("Destroying Stargate...")
    # Placeholder for destroying a Stargate
    subprocess.run(["echo", "Stargate destroyed."])


if __name__ == "__main__":
    main()

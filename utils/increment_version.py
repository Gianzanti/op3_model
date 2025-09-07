import subprocess

import semver
import toml

# Path to your pyproject.toml file
PYPROJECT_PATH = "pyproject.toml"

if __name__ == "__main__":
    try:
        # Load the pyproject.toml file
        with open(PYPROJECT_PATH, "r") as f:
            pyproject = toml.load(f)

        # Get the current version
        if "project" in pyproject and "version" in pyproject["project"]:
            current_version = pyproject["project"]["version"]
            print(f"Current version: {current_version}")
        else:
            raise ValueError("Version key not found in pyproject.toml")  #

        # Increment the version (e.g., patch version)
        new_version = semver.VersionInfo.parse(current_version).bump_patch()
        print(f"Incremented version: {new_version}")

        # Update the version in the pyproject.toml file
        pyproject["project"]["version"] = str(new_version)

        # Write the updated pyproject.toml file
        with open(PYPROJECT_PATH, "w") as f:
            toml.dump(pyproject, f)

        print(f"Version incremented to {new_version}")

        # Stage the modified pyproject.toml file
        subprocess.run(["git", "add", PYPROJECT_PATH], check=True)
        print(f"Staged {PYPROJECT_PATH} for commit.")

    except FileNotFoundError:
        print("Error: pyproject.toml not found.")

    except ValueError as e:  # Catch the exception raised above
        print(f"Error: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")

import os

VERSION_FILE_NAME: str = "version.txt"
COMMIT_ENV: str = "GITHUB_COMMIT"

if __name__ == "__main__":
    print("Writing version to file ...")
    if commit := os.environ.get(COMMIT_ENV):
        with open(VERSION_FILE_NAME, "w") as f:
            f.write(commit)
    else:
        print("Commit hash not found")

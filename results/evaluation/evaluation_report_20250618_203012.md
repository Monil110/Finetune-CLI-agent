# Model Evaluation Report
Generated on: 2025-06-18 20:30:12

## Summary Statistics

**Base Model:**
- Average Plan Quality: 1.00 / 2.0
- Average Commands per Response: 5.4

**Fine-tuned Model:**
- Average Plan Quality: 1.14 / 2.0
- Average Commands per Response: 5.3

**Improvements:**
- Plan Quality: +0.14
- Commands Count: -0.1

**Similarity Metrics (Base vs Fine-tuned):**
- BLEU Score: 0.146
- ROUGE-1: 0.443
- ROUGE-L: 0.330

## Individual Results

### Prompt 1
**Instruction:** Create a new Git repository and make an initial commit

**Base Model Response:**
```
1. Open your terminal or command prompt and navigate to the directory where you want to create a new git repository.
2. Use the `git init` command to initialize the new repository:
   ```
   $ git init
   Initialized empty Git repository in /path/to/new-repo/.git/
   Your branch is up to date with 'origin/master'.
   The default branch is set to master.
   To add branches, use 'git branch [branch name]' (e.g., 'git checkout -b my_branch').
   You can also add remote repositories by
```
- Plan Quality: 1/2
- Commands Extracted: 7

**Fine-tuned Model Response:**
```
1. Open your terminal or command prompt on your computer.
2. Create a new directory for your project using the `mkdir` command. For example:
   ```bash
   mkdir my_project
   cd my_projects/my_project_name
   touch README.md
   git init
   echo "This is a sample READMe file" > READ ME.md # Add some text to the READ Me file
   commit -m "Initial commit" # Commit the changes made in the previous step
   push origin master # Push the changes to the remote
```
- Plan Quality: 1/2
- Commands Extracted: 9
**Metrics:**
- BLEU Score: 0.117
- ROUGE-1: 0.400
- ROUGE-2: 0.152
- ROUGE-L: 0.287

---

### Prompt 2
**Instruction:** Set up a Python virtual environment and install pandas

**Base Model Response:**
```
1. Open your terminal or command prompt (Mac/Linux)
2. Create a new directory for your project
   ```
   mkdir my_project
   cd my_projec
   python -m venv env
   source env/bin/activate
   pip install --upgrade pip
   # Install pandas if it's not already installed
   sudo apt-get update && sudo apt upgrade
   curl https://bootstrap.pypa.io/get-pip.py | python
   export PIP_NO_CACHE=true
   echo "export PIPENV
```
- Plan Quality: 2/2
- Commands Extracted: 11

**Fine-tuned Model Response:**
```
1. Open your terminal or command prompt and create a new directory for your project. For example, if you're working on a Python project named "my_project", create a directory called "my-project".

2. Navigate to the newly created directory using the command `cd my-project`.

3. Create a new virtual environment by running the following command in your terminal:

   ```bash
   python -m venv env
   source env/bin/activate
   pip install --upgrade pip setuptools wheel
   # Install Pandas as a dependency
```
- Plan Quality: 2/2
- Commands Extracted: 6
**Metrics:**
- BLEU Score: 0.271
- ROUGE-1: 0.473
- ROUGE-2: 0.356
- ROUGE-L: 0.432

---

### Prompt 3
**Instruction:** Find all Python files in the current directory that contain 'import numpy'

**Base Model Response:**
```
1. Open your favorite text editor (such as VS Code or Sublime Text) and navigate to the directory where you want to search for the `import numpy` statement.
2. Type `cd [directory_path]` into the command line to change to the desired directory.
3. Use the `find` command to search through all files in that directory for any file ending in `.py`. This will find all Python scripts in the directory. For example:
   ```
   find . -name "*.py"
   # Output:
   ├── __init__.py
   └── main.py
```
- Plan Quality: 1/2
- Commands Extracted: 4

**Fine-tuned Model Response:**
```
1. Open your favorite text editor (e.g., VSCode or Sublime Text) and navigate to the directory containing the Python files you want to search for the import statement.
2. Type `cd` followed by the directory path (either absolute or relative). For example:
   ```
   cd /path/to/directory/with/python/files
   # or
   mkdir -p /path_to_directory_containing_python_files && cd /home/user/path_of_directory/containing/python_file
   python -m venv env
```
- Plan Quality: 1/2
- Commands Extracted: 5
**Metrics:**
- BLEU Score: 0.212
- ROUGE-1: 0.510
- ROUGE-2: 0.318
- ROUGE-L: 0.444

---

### Prompt 4
**Instruction:** Create a compressed archive of all .txt files in the current directory

**Base Model Response:**
```
1. Open a terminal or command prompt window on your computer.
2. Navigate to the directory where you want to create the compressed file. For example, if you're in the "Documents" folder, type:
   ```
   cd Documents
   mkdir Compressed_Files
   echo "Creating Compressed Files Directory..."
   touch CompressedFiles/CompressedFiles.txt
   ls -lh CompressedFile*
   exit
   `

3. Create a new text file named "Compressed_files.txt". This will contain a list of all the
```
- Plan Quality: 1/2
- Commands Extracted: 8

**Fine-tuned Model Response:**
```
1. Open your terminal or command prompt and navigate to the directory containing the .txt file(s) you want to compress.
2. Use the `ls` command to list all the text files in that directory. You can use wildcards to include subdirectories as well:
   ```
   ls *.txt
   # Output:
   ├── index.html
   └── README.md
   ...
   [...]
   `-- 100%`
   Elapsed Time: 0:00:14
   Total Files: 37
   Av
```
- Plan Quality: 1/2
- Commands Extracted: 5
**Metrics:**
- BLEU Score: 0.110
- ROUGE-1: 0.489
- ROUGE-2: 0.171
- ROUGE-L: 0.290

---

### Prompt 5
**Instruction:** List all files modified in the last 24 hours, sorted by modification time

**Base Model Response:**
```
1. First, make sure you have the latest version of your Bash or Zsh installation. You can check if you have it by running `bash --version` or `zsh -v`.

2. Next, create a new file called `last_modified_files.txt` and save it in your home directory. This will be the output file for the script.

3. Open a terminal window and navigate to your home folder.
   ```bash
   $ cd ~
   # or
   > cd /home/your_username
   (replace "your_user" with your
```
- Plan Quality: 1/2
- Commands Extracted: 1

**Fine-tuned Model Response:**
```
1. Open your terminal or command prompt and navigate to the directory where you want to list the modified files.
2. Use the `ls` command to list all files in that directory (e.g., `ls -l`). This will display the file names, their sizes, and the date they were created/modified.
3. To filter the output to only show modified files, use the `-mtime` option:
   ```
   ls -m -l | grep '^d' | awk '{print $5}' | sort -n | head -n 10
   # Output:
```
- Plan Quality: 1/2
- Commands Extracted: 4
**Metrics:**
- BLEU Score: 0.021
- ROUGE-1: 0.316
- ROUGE-2: 0.064
- ROUGE-L: 0.190

---

### Prompt 6
**Instruction:** Recursively search for files larger than 100MB and move them to a backup directory

**Base Model Response:**
```
To recursively search and move files larger that 1 MB (1024 bytes) to a specific backup directory, you can use the following shell script:

```bash
#!/bin/bash

# Set up variables
backup_dir="/path/to/backup"
file_ext=".zip"

if [[ -d $backupDir ]]; then
    echo "Backup directory already exists."
else
    mkdir -p $backuppath
fi

for file in $(find $backuptmp -type f -size +104857
```
- Plan Quality: 1/2
- Commands Extracted: 5

**Fine-tuned Model Response:**
```
To recursively search and move files larger then 1 MB to a new backup directory using the find command in Bash, you can use the following steps:

1. Create a new directory where you want to store your backup files. For example, let's say you want the backup directory to be called "backup".

2. Set up the environment variables for the current working directory (CWD) and the backup destination directory. You can do this by running the following commands:
   ```bash
   export CWD="$(pwd)"
   mkdir -p "$(dirname "$BACKUP_
```
- Plan Quality: 1/2
- Commands Extracted: 5
**Metrics:**
- BLEU Score: 0.137
- ROUGE-1: 0.470
- ROUGE-2: 0.245
- ROUGE-L: 0.336

---

### Prompt 7
**Instruction:** Set up a cron job to automatically backup a directory every hour using tar and gzip

**Base Model Response:**
```
1. First, create a new file called "backup_schedule" in your home directory with the following content:
```
0 2 * * * /path/to/backup/script.sh
``` 
This will run the script at midnight on the second day of every month.

2. Create a new crontab file in your crontabs folder (usually located at `/etc/cron.d/` or `/var/spool/crons` depending on your system). Open it with a text editor and add the following lines:

```crontab
```
- Plan Quality: 0/2
- Commands Extracted: 2

**Fine-tuned Model Response:**
```
1. First, you need to create a new file called "backup_schedule.sh" in your home directory (e.g., /home/username). This script will be used as a cronjob.

2. Open the "backups/backup.sh", which is the main script that will be executed by the cronjob, and add the following lines:

```bash
#!/bin/bash

# Set the backup schedule
if [ "$1" == "hourly" ]; then
  # Backup every hour
  echo "Backing up every hour..."
```
- Plan Quality: 1/2
- Commands Extracted: 3
**Metrics:**
- BLEU Score: 0.154
- ROUGE-1: 0.441
- ROUGE-2: 0.196
- ROUGE-L: 0.331

---

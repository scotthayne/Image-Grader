import subprocess
import os

def run_exiftool_test(target_file):
    """
    Runs a detailed test of the ExifTool command and prints all output.
    """
    print("--- Running ExifTool Detailed Test ---")
    
    if not os.path.exists(target_file):
        print(f"Error: Target file not found at '{target_file}'")
        return

    print(f"Target File: {target_file}")
    
    # Command to set the XMP Rating tag to 5
    command = ['exiftool', '-overwrite_original', '-Rating=5', target_file]
    
    print(f"Executing command: {' '.join(command)}")
    print("-" * 40)

    try:
        # Execute the command and capture all output
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=False  # Set to False to prevent raising an error, so we can inspect the output
        )

        print("--- ExifTool Standard Output ---")
        if result.stdout:
            print(result.stdout)
        else:
            print("(No standard output)")
        
        print("\n--- ExifTool Standard Error ---")
        if result.stderr:
            print(result.stderr)
        else:
            print("(No standard error)")
            
        print("\n--- Test Summary ---")
        if result.returncode == 0:
            print("Command executed successfully (Exit Code 0).")
            print("Check the file's properties to see if the rating was applied.")
        else:
            print(f"Command failed with Exit Code {result.returncode}.")

    except FileNotFoundError:
        print("FATAL ERROR: 'exiftool' command not found.")
        print("Please ensure exiftool.exe is installed and its location is in the system's PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_file_path = "D:\\Memex RAWS\\495A9327.CR2"
    run_exiftool_test(test_file_path)

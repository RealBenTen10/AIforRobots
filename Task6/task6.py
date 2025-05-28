import subprocess
import time

# Command to run
command = ["ros2", "topic", "echo", "/ir_intensity"]

# Open the output file
with open("data.txt", "w") as output_file:
    # Start the subprocess
    process = subprocess.Popen(command, stdout=output_file, stderr=subprocess.STDOUT)

    try:
        # Let it run for 10 seconds
        time.sleep(10)
    finally:
        # Terminate the process
        process.terminate()
        # Wait for it to fully exit
        process.wait()

print("Data collection stopped after 10 seconds.")

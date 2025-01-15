import serial
import time

# Replace 'COM5' with the correct port for your Arduino (e.g., 'COM3' on Windows, '/dev/ttyACM0' on Linux, etc.)
SERIAL_PORT = "COM5"
BAUD_RATE = 9600


def main():
    # Open serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Give Arduino some time to reset after opening the port

    # The command we want to send
    test_command = "3,0,0\n"

    # Send it
    ser.write(test_command.encode("utf-8"))
    print("Sent to Arduino:", test_command.strip())

    # Continuously read and print anything Arduino sends back
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if line:
                print("Arduino says:", line)


if __name__ == "__main__":
    main()
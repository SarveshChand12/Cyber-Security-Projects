import socket
import threading
import queue
import argparse
import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import ipaddress
import time

# Configure logging
logging.basicConfig(
    filename='port_scanner.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

# Define a dictionary of common ports and their associated services
COMMON_SERVICES = {
    20: 'FTP (Data Transfer)',
    21: 'FTP (Control)',
    22: 'SSH',
    23: 'Telnet',
    25: 'SMTP',
    53: 'DNS',
    80: 'HTTP',
    110: 'POP3',
    111: 'RPCbind',
    135: 'Microsoft EPMAP',
    139: 'NetBIOS',
    143: 'IMAP',
    443: 'HTTPS',
    445: 'Microsoft-DS',
    993: 'IMAPS',
    995: 'POP3S',
    1723: 'PPTP',
    3306: 'MySQL',
    3389: 'RDP',
    5900: 'VNC',
    8080: 'HTTP Proxy',
    # Add more ports and services as needed
}

# Placeholder for advanced service detection logic or integration with external databases
def advanced_service_detection(port, banner):
    # Example: Analyze banner or use external databases/APIs
    # For simplicity, we'll return the common service or banner info
    if port in COMMON_SERVICES:
        return COMMON_SERVICES[port]
    elif banner:
        return f"Service: {banner.strip()}"
    else:
        return "Unknown Service"

# Function to resolve hostname to IP (supports IPv4 and IPv6)
def resolve_target(target):
    try:
        # Attempt to parse as IP address
        ip_obj = ipaddress.ip_address(target)
        return str(ip_obj)
    except ValueError:
        try:
            # Resolve hostname to IP
            addr_info = socket.getaddrinfo(target, None)
            ip = addr_info[0][4][0]
            return ip
        except socket.gaierror:
            logging.error(f"Unable to resolve host '{target}'.")
            return None

# Worker function for each thread
def port_worker(q, target_ip, open_ports, timeout, semaphore):
    while not q.empty():
        port = q.get()
        try:
            with semaphore:
                sock = socket.socket(socket.AF_INET6 if ':' in target_ip else socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((target_ip, port))
                if result == 0:
                    try:
                        # Banner Grabbing
                        sock.sendall(b'HEAD / HTTP/1.0\r\n\r\n')
                        banner = sock.recv(1024).decode().strip()
                    except socket.error:
                        banner = ''
                    service = advanced_service_detection(port, banner)
                    open_ports.append((port, service))
                sock.close()
        except Exception as e:
            logging.error(f"Error scanning port {port}: {e}")
        finally:
            q.task_done()

# Function to scan ports with threading and rate limiting
def scan_ports(target, start_port, end_port, num_threads=100, timeout=1.0, rate_limit=100):
    target_ip = resolve_target(target)
    if not target_ip:
        return None, "Invalid target."

    print(f"Scanning Target: {target} ({target_ip})")
    print(f"Port Range: {start_port} to {end_port}")
    print(f"Using {num_threads} threads with rate limit {rate_limit} connections.\n")

    # Create a queue and populate it with ports to scan
    port_queue = queue.Queue()
    for port in range(start_port, end_port + 1):
        port_queue.put(port)

    open_ports = []
    threads = []
    semaphore = threading.Semaphore(rate_limit)  # Rate limiting

    # Start threads
    for _ in range(num_threads):
        thread = threading.Thread(target=port_worker, args=(port_queue, target_ip, open_ports, timeout, semaphore))
        thread.daemon = True  # Allows threads to exit even if still running
        thread.start()
        threads.append(thread)

    # Wait for all ports to be scanned
    port_queue.join()

    # Sort open ports
    open_ports = sorted(open_ports, key=lambda x: x[0])

    return open_ports, None

# Function to parse command-line arguments (not used in GUI version but kept for completeness)
def parse_arguments():
    parser = argparse.ArgumentParser(description="Enhanced Port Scanner with GUI")
    parser.add_argument("target", help="Target IP address or hostname to scan")
    parser.add_argument("-s", "--start", type=int, default=1, help="Start of port range (default: 1)")
    parser.add_argument("-e", "--end", type=int, default=1024, help="End of port range (default: 1024)")
    parser.add_argument("-t", "--threads", type=int, default=100, help="Number of concurrent threads (default: 100)")
    parser.add_argument("-rl", "--rate_limit", type=int, default=100, help="Maximum simultaneous connections (default: 100)")
    parser.add_argument("-to", "--timeout", type=float, default=1.0, help="Timeout for socket connections in seconds (default: 1.0)")
    parser.add_argument("-o", "--output", help="Output file to save results")
    args = parser.parse_args()

    # Validate port numbers
    if args.start < 1 or args.start > 65535:
        parser.error("Start port must be between 1 and 65535.")
    if args.end < 1 or args.end > 65535:
        parser.error("End port must be between 1 and 65535.")
    if args.start > args.end:
        parser.error("Start port must be less than or equal to end port.")

    return args

# GUI Application Class
class PortScannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Port Scanner")
        self.create_widgets()
        self.open_ports = []
        self.is_scanning = False

    def create_widgets(self):
        # Target Frame
        target_frame = ttk.LabelFrame(self.root, text="Target Specification")
        target_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(target_frame, text="IP Address / Hostname:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.target_entry = ttk.Entry(target_frame, width=30)
        self.target_entry.grid(row=0, column=1, padx=5, pady=5)

        # Port Range Frame
        port_frame = ttk.LabelFrame(self.root, text="Port Range")
        port_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(port_frame, text="Start Port:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.start_port_entry = ttk.Entry(port_frame, width=10)
        self.start_port_entry.grid(row=0, column=1, padx=5, pady=5)
        self.start_port_entry.insert(0, "1")

        ttk.Label(port_frame, text="End Port:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.end_port_entry = ttk.Entry(port_frame, width=10)
        self.end_port_entry.grid(row=0, column=3, padx=5, pady=5)
        self.end_port_entry.insert(0, "1024")

        # Scan Options Frame
        options_frame = ttk.LabelFrame(self.root, text="Scan Options")
        options_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(options_frame, text="Threads:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.threads_entry = ttk.Entry(options_frame, width=10)
        self.threads_entry.grid(row=0, column=1, padx=5, pady=5)
        self.threads_entry.insert(0, "100")

        ttk.Label(options_frame, text="Rate Limit:").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.rate_limit_entry = ttk.Entry(options_frame, width=10)
        self.rate_limit_entry.grid(row=0, column=3, padx=5, pady=5)
        self.rate_limit_entry.insert(0, "100")

        ttk.Label(options_frame, text="Timeout (s):").grid(row=0, column=4, sticky="w", padx=5, pady=5)
        self.timeout_entry = ttk.Entry(options_frame, width=10)
        self.timeout_entry.grid(row=0, column=5, padx=5, pady=5)
        self.timeout_entry.insert(0, "1.0")

        # Output Frame
        output_frame = ttk.LabelFrame(self.root, text="Output")
        output_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        ttk.Label(output_frame, text="Save to File:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.output_entry = ttk.Entry(output_frame, width=30)
        self.output_entry.grid(row=0, column=1, padx=5, pady=5)

        # Scan Button
        self.scan_button = ttk.Button(self.root, text="Start Scan", command=self.start_scan)
        self.scan_button.grid(row=4, column=0, padx=10, pady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(self.root, orient='horizontal', length=400, mode='determinate')
        self.progress.grid(row=5, column=0, padx=10, pady=10)

        # Results Frame
        results_frame = ttk.LabelFrame(self.root, text="Scan Results")
        results_frame.grid(row=6, column=0, padx=10, pady=10, sticky="nsew")

        self.results_text = tk.Text(results_frame, height=15, width=80, state='disabled')
        self.results_text.grid(row=0, column=0, padx=5, pady=5)

        # Configure grid weights
        self.root.grid_rowconfigure(6, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)

    def start_scan(self):
        if self.is_scanning:
            messagebox.showwarning("Scan in Progress", "A scan is already in progress.")
            return

        target = self.target_entry.get().strip()
        start_port = self.start_port_entry.get().strip()
        end_port = self.end_port_entry.get().strip()
        threads = self.threads_entry.get().strip()
        rate_limit = self.rate_limit_entry.get().strip()
        timeout = self.timeout_entry.get().strip()
        output_file = self.output_entry.get().strip()

        # Input validation
        if not target:
            messagebox.showerror("Input Error", "Please enter a target IP address or hostname.")
            return
        try:
            start_port = int(start_port)
            end_port = int(end_port)
            threads = int(threads)
            rate_limit = int(rate_limit)
            timeout = float(timeout)
            if start_port < 1 or start_port > 65535 or end_port < 1 or end_port > 65535:
                raise ValueError
            if start_port > end_port:
                messagebox.showerror("Input Error", "Start port must be less than or equal to end port.")
                return
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for ports, threads, rate limit, and timeout.")
            return

        # Disable scan button
        self.scan_button.config(state='disabled')
        self.is_scanning = True

        # Clear previous results
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state='disabled')

        # Start scanning in a separate thread to keep the GUI responsive
        scan_thread = threading.Thread(
            target=self.perform_scan,
            args=(target, start_port, end_port, threads, rate_limit, timeout, output_file)
        )
        scan_thread.start()

    def perform_scan(self, target, start_port, end_port, threads, rate_limit, timeout, output_file):
        try:
            total_ports = end_port - start_port + 1
            self.progress['maximum'] = total_ports
            open_ports, error = scan_ports(target, start_port, end_port, threads, timeout, rate_limit)
            if error:
                messagebox.showerror("Scan Error", error)
                self.reset_scan()
                return

            # Update results
            self.results_text.config(state='normal')
            if open_ports:
                result_str = f"Open Ports for {target}:\n"
                for port, service in open_ports:
                    result_str += f"Port {port}: {service}\n"
                self.results_text.insert(tk.END, result_str)
                logging.info(f"Scan Results for {target}: {open_ports}")
            else:
                self.results_text.insert(tk.END, "No open ports found in the specified range.")
                logging.info(f"No open ports found for {target} in range {start_port}-{end_port}.")

            self.results_text.config(state='disabled')

            # Save to file if specified
            if output_file:
                try:
                    with open(output_file, 'w') as f:
                        if open_ports:
                            f.write(f"Open Ports for {target}:\n")
                            for port, service in open_ports:
                                f.write(f"Port {port}: {service}\n")
                        else:
                            f.write(f"No open ports found for {target} in the range {start_port}-{end_port}.\n")
                    logging.info(f"Results saved to {output_file}")
                except IOError as e:
                    logging.error(f"Unable to write to file '{output_file}': {e}")
                    messagebox.showerror("File Error", f"Unable to write to file '{output_file}'.")
        except Exception as e:
            logging.error(f"Unexpected error during scan: {e}")
            messagebox.showerror("Scan Error", f"An unexpected error occurred: {e}")
        finally:
            self.reset_scan()

    def reset_scan(self):
        self.is_scanning = False
        self.scan_button.config(state='normal')
        self.progress['value'] = 0

# Main function to run the GUI
def main():
    root = tk.Tk()
    app = PortScannerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

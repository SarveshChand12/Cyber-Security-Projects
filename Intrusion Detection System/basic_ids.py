#!/usr/bin/env python3
"""
Basic Intrusion Detection System (IDS) - Advanced Example
Author: Your Name
Description:
    A simple IDS using Scapy to capture packets and detect suspicious activity
    based on predefined rules. Alerts are logged and optionally displayed in real time.

Usage:
    sudo python basic_ids.py [-i <interface>] [-r <rules_file>] [-l <log_file>]
    
    Example:
        sudo python basic_ids.py -i eth0 -r rules.json -l ids_log.txt
"""

import argparse
import time
import json
import logging
from scapy.all import (
    sniff, 
    TCP, 
    UDP, 
    IP,
    Ether, 
    Raw,
    conf
)

# ======================= Sample Rules Data Structure =======================
# Example: "rules.json"
# {
#     "known_bad_ips": ["192.168.10.10"],
#     "blocked_ports": [23, 445, 3389],
#     "scan_threshold": 10, 
#     "scan_time_window": 5
# }
#
# Explanation:
# - known_bad_ips: Any traffic from/to these IPs triggers an alert.
# - blocked_ports: Accessing these ports triggers an alert (e.g., Telnet-23, SMB-445, RDP-3389).
# - scan_threshold & scan_time_window: If a single source IP sends more than scan_threshold
#   distinct port probes within scan_time_window seconds, trigger a port scan alert.
# ==========================================================================

class IDSConfig:
    """
    Configuration container, holding:
    - interface to listen on,
    - rules data,
    - logging settings,
    - additional parameters for the detection engine.
    """
    def __init__(self, interface, rules, log_file):
        self.interface = interface
        self.rules = rules
        self.log_file = log_file

class IDSRuleEngine:
    """Class that applies rules to captured packets and determines if alerts should be triggered."""
    def __init__(self, config):
        self.config = config

        # Parse rules from the config
        self.known_bad_ips = set(self.config.rules.get("known_bad_ips", []))
        self.blocked_ports = set(self.config.rules.get("blocked_ports", []))
        self.scan_threshold = self.config.rules.get("scan_threshold", 10)
        self.scan_time_window = self.config.rules.get("scan_time_window", 5)

        # Data structure to track possible port scans:
        # {source_ip: [(timestamp, dst_port), ...]}
        self.scan_tracker = {}

    def evaluate_packet(self, pkt):
        """Evaluate a single packet against the rules. Return a list of alerts (if any)."""
        alerts = []

        if IP in pkt:
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = None
            dst_port = None

            # Check known bad IPs
            if src_ip in self.known_bad_ips or dst_ip in self.known_bad_ips:
                alerts.append(f"Traffic involving known bad IP {src_ip} -> {dst_ip}")

            # Check for TCP or UDP to extract port info
            if TCP in pkt:
                proto = "TCP"
                dst_port = pkt[TCP].dport
            elif UDP in pkt:
                proto = "UDP"
                dst_port = pkt[UDP].dport

            # If we have a port, check blocked ports
            if dst_port and dst_port in self.blocked_ports:
                alerts.append(f"Access attempt to blocked port {dst_port} ({proto}), from {src_ip}")

            # Potential port scan detection
            if dst_port and proto == "TCP":
                # Track distinct ports visited by a given source within time window
                current_time = time.time()
                self._track_port_scan(src_ip, dst_port, current_time)
                # Evaluate if a scan threshold is exceeded
                if self._is_port_scan(src_ip, current_time):
                    alerts.append(f"Possible port scan from {src_ip} exceeding {self.scan_threshold} ports in {self.scan_time_window} seconds")

        return alerts

    def _track_port_scan(self, src_ip, dst_port, timestamp):
        """Add a (timestamp, port) entry for a source IP in the scan tracker."""
        if src_ip not in self.scan_tracker:
            self.scan_tracker[src_ip] = []
        self.scan_tracker[src_ip].append((timestamp, dst_port))

        # Clean up old entries beyond the time window
        valid_time = timestamp - self.scan_time_window
        self.scan_tracker[src_ip] = [
            (t, p) for (t, p) in self.scan_tracker[src_ip] if t >= valid_time
        ]

    def _is_port_scan(self, src_ip, current_time):
        """
        Check if the number of distinct ports visited by src_ip
        within the last scan_time_window seconds is above scan_threshold.
        """
        if src_ip not in self.scan_tracker:
            return False
        distinct_ports = {p for (t, p) in self.scan_tracker[src_ip]}
        if len(distinct_ports) > self.scan_threshold:
            return True
        return False

class IDSLogger:
    """Handles logging of alerts, either to file or console, and optional additional alerting."""
    def __init__(self, log_file):
        self.log_file = log_file
        # Configure Python's built-in logging
        logging.basicConfig(
            filename=log_file,
            filemode='a',
            format='%(asctime)s [%(levelname)s] %(message)s',
            level=logging.INFO
        )

    def log_alerts(self, alerts):
        for alert in alerts:
            # Log to file
            logging.warning(alert)
            # Print to console (optional)
            print(f"[ALERT] {alert}")

    def log_info(self, message):
        logging.info(message)
        print(message)

class BasicIDS:
    """
    The main IDS class that:
    - Initializes the rule engine and logger.
    - Captures packets via Scapy.
    - Processes each packet, logs alerts, etc.
    """
    def __init__(self, config):
        self.config = config
        self.rule_engine = IDSRuleEngine(config)
        self.logger = IDSLogger(config.log_file)

    def start(self):
        """Begin packet capture and intrusion detection."""
        self.logger.log_info(f"Starting IDS on interface '{self.config.interface}'...")
        self.logger.log_info("Press Ctrl+C to stop.")

        # Sniff packets
        sniff(
            iface=self.config.interface,
            prn=self._process_packet,
            store=False,
            filter="ip"  # Basic filter to capture only IP-based traffic
        )

    def _process_packet(self, pkt):
        """Callback for each captured packet."""
        alerts = self.rule_engine.evaluate_packet(pkt)
        if alerts:
            self.logger.log_alerts(alerts)

# ========================= Command Line Interface ===========================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Basic Intrusion Detection System (IDS)")
    parser.add_argument("-i", "--interface", default=None, help="Network interface to monitor (e.g., eth0).")
    parser.add_argument("-r", "--rules", default=None, help="Path to rules JSON file.")
    parser.add_argument("-l", "--log_file", default="ids_log.txt", help="Path to the log file.")
    return parser.parse_args()

def load_rules(rules_file):
    """Load IDS rules from a JSON file."""
    if rules_file is None:
        # Default minimal rules if no file is provided
        return {
            "known_bad_ips": ["10.10.10.10"],
            "blocked_ports": [23, 445, 3389],
            "scan_threshold": 5,
            "scan_time_window": 5
        }
    try:
        with open(rules_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Failed to load rules file: {e}")
        # Fallback to default
        return {
            "known_bad_ips": ["10.10.10.10"],
            "blocked_ports": [23, 445, 3389],
            "scan_threshold": 5,
            "scan_time_window": 5
        }

def main():
    args = parse_arguments()
    rules = load_rules(args.rules)

    # If interface not specified, Scapy tries to find an appropriate interface
    interface = args.interface if args.interface else conf.iface

    # Create IDS config
    config = IDSConfig(
        interface=interface,
        rules=rules,
        log_file=args.log_file
    )

    # Initialize and start IDS
    ids = BasicIDS(config)
    try:
        ids.start()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping IDS.")
    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")

if __name__ == "__main__":
    main()

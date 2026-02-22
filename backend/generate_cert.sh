#!/bin/bash

# Generate a self-signed certificate
echo "🔒 Generating self-signed certificate..."

# Define file names
KEY_FILE="key.pem"
CERT_FILE="cert.pem"

# Check if certificates already exist
if [ -f "$KEY_FILE" ] && [ -f "$CERT_FILE" ]; then
    echo "✅ Certificates already exist. Skipping generation."
    exit 0
fi

# Generate key and cert
# Subject /C=US/ST=State/L=City/O=Organization/CN=localhost
openssl req -x509 -newkey rsa:4096 -keyout "$KEY_FILE" -out "$CERT_FILE" -days 365 -nodes -subj "/C=US/ST=Dev/L=Dev/O=Dev/CN=localhost"

echo "✅ Generated $KEY_FILE and $CERT_FILE"

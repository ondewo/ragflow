#!/bin/sh
# Creates a throwaway CA and one certificate per name for local development: RAGFlow connects to OpenSearch, Redis and S3 only over TLS with a client certificate.
# Usage: docker/generate_dev_certs.sh [name ...]   (default: ragflow redis opensearch01)
# Writes docker/certs/<name>/{ca.pem,cert.pem,key.pem}; each certificate is valid as server and client for <name>, localhost and 127.0.0.1. Never use them in production.
set -eu

dir="$(cd "$(dirname "$0")" && pwd)/certs"
[ "$#" -gt 0 ] || set -- ragflow redis opensearch01
mkdir -p "$dir/ca"

if [ ! -f "$dir/ca/ca.pem" ]; then
    openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out "$dir/ca/ca.key"
    openssl req -x509 -new -key "$dir/ca/ca.key" -sha256 -days 825 -subj "/CN=RAGFlow development CA" -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign,cRLSign" -out "$dir/ca/ca.pem"
fi

for name in "$@"; do
    mkdir -p "$dir/$name"
    openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out "$dir/$name/key.pem"
    openssl req -new -key "$dir/$name/key.pem" -subj "/CN=$name" -out "$dir/$name/cert.csr"
    printf 'subjectAltName=DNS:%s,DNS:localhost,IP:127.0.0.1\nextendedKeyUsage=serverAuth,clientAuth\nkeyUsage=critical,digitalSignature,keyEncipherment\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid\n' "$name" > "$dir/$name/cert.ext"
    openssl x509 -req -in "$dir/$name/cert.csr" -CA "$dir/ca/ca.pem" -CAkey "$dir/ca/ca.key" -CAcreateserial -sha256 -days 825 -extfile "$dir/$name/cert.ext" -out "$dir/$name/cert.pem"
    rm -f "$dir/$name/cert.csr" "$dir/$name/cert.ext"
    cp "$dir/ca/ca.pem" "$dir/$name/ca.pem"
    # Readable by the non-root users of the Redis and OpenSearch images.
    chmod 0644 "$dir/$name/key.pem"
    echo "Wrote $dir/$name"
done

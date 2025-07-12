#!/usr/bin/env bash
# Script para capturar tráfego “normal” da rede doméstica
INTERFACE="enp3s0"      # Substitua pelo nome da sua interface
OUTPUT="data/raw/normal.pcap"
DURATION=300            # Duração em segundos (5 minutos)

echo "Capturando tráfego na interface $INTERFACE por $DURATION segundos..."
sudo timeout $DURATION tcpdump -i $INTERFACE -w $OUTPUT
echo "Captura concluída: $OUTPUT"

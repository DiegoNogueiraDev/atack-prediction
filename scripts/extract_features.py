#!/usr/bin/env python3
# scripts/extract_features.py

import logging
import pyshark
import pandas as pd

# Configuração básica de logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)

# Intervalo para logs de progresso (em número de pacotes)
LOG_INTERVAL = 1000

def pcap_to_flows(pcap_path):
    """
    Lê um .pcap e retorna um DataFrame de fluxos com features:
      - bytes totais
      - número de pacotes
      - duração do fluxo
      - mean/std de inter-arrival
    """
    logging.info(f"▶ Iniciando leitura de {pcap_path}")
    cap = pyshark.FileCapture(pcap_path, keep_packets=False)

    flows = {}
    packet_count = 0
    
    for pkt in cap:
        packet_count += 1
        
        try:
            # Tenta obter endereços IP
            if hasattr(pkt, 'ip'):
                src = pkt.ip.src
                dst = pkt.ip.dst
            elif hasattr(pkt, 'ipv6'):
                src = pkt.ipv6.src
                dst = pkt.ipv6.dst
            else:
                continue
                
            proto = pkt.highest_layer
            sport = dport = None
            
            # Tenta obter portas se for TCP/UDP
            if hasattr(pkt, 'tcp'):
                sport = pkt.tcp.srcport
                dport = pkt.tcp.dstport
            elif hasattr(pkt, 'udp'):
                sport = pkt.udp.srcport
                dport = pkt.udp.dstport

            key = (src, dst, proto, sport, dport)
            info = flows.setdefault(key, {'bytes': 0, 'pkts': 0, 'times': []})
            info['bytes'] += int(pkt.length)
            info['pkts'] += 1
            info['times'].append(float(pkt.sniff_time.timestamp()))

            # Log de progresso a cada LOG_INTERVAL pacotes
            if packet_count % LOG_INTERVAL == 0:
                logging.info(f"    {packet_count} pacotes processados em {pcap_path}")
                
        except Exception as e:
            logging.debug(f"Erro processando pacote {packet_count}: {e}")
            continue

    cap.close()
    logging.info(f"✔ Leitura concluída de {pcap_path}, fluxos capturados: {len(flows)}")

    records = []
    for (src, dst, proto, sport, dport), v in flows.items():
        times = sorted(v['times'])
        iat = [t2 - t1 for t1, t2 in zip(times, times[1:])] if len(times) > 1 else [0]
        records.append({
            'src':       src,
            'dst':       dst,
            'proto':     proto,
            'sport':     sport or '',
            'dport':     dport or '',
            'bytes':     v['bytes'],
            'pkts':      v['pkts'],
            'duration':  times[-1] - times[0],
            'iat_mean':  sum(iat) / len(iat),
            'iat_std':   pd.Series(iat).std()
        })

    return pd.DataFrame(records)


if __name__ == '__main__':
    logging.info("=== Extraindo features de normal.pcap ===")
    df_norm = pcap_to_flows('data/raw/normal.pcap')
    df_norm['label'] = 0

    logging.info("=== Extraindo features de attack.pcap ===")
    df_att  = pcap_to_flows('data/raw/attack.pcap')
    df_att['label'] = 1

    # Concatena e salva
    df_all = pd.concat([df_norm, df_att], ignore_index=True)
    df_all.to_csv('data/processed/flows.csv', index=False)
    logging.info(f"🎉 Features extraídas: {df_all.shape[0]} fluxos → data/processed/flows.csv")

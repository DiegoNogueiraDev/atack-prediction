# scripts/extract_features.py

import pyshark
import pandas as pd

def pcap_to_flows(pcap_path):
    """
    Lê um .pcap e retorna um DataFrame de fluxos com features:
    - bytes totais
    - número de pacotes
    - duração do fluxo
    - mean/std de inter-arrival
    """
    cap = pyshark.FileCapture(
        pcap_path,
        only_summaries=True,   # resumo leve por pacote
        keep_packets=False     # não guarda tudo na memória
    )

    flows = {}
    for pkt in cap:
        # Chave do fluxo: (src, dst, protocolo, srcport, dstport)
        fields = pkt._all_fields
        try:
            src = fields['Layer 2'].src
            dst = fields['Layer 2'].dst
            proto = pkt.protocol
            sport = fields.get(pkt.transport_layer + '.srcport', None)
            dport = fields.get(pkt.transport_layer + '.dstport', None)
        except Exception:
            continue

        key = (src, dst, proto, sport, dport)
        info = flows.setdefault(key, {'bytes': 0, 'pkts': 0, 'times': []})
        info['bytes'] += int(pkt.length)
        info['pkts'] += 1
        info['times'].append(float(pkt.time))

    cap.close()

    records = []
    for (src, dst, proto, sport, dport), v in flows.items():
        times = sorted(v['times'])
        # cálculo de inter-arrival times
        iat = [t2 - t1 for t1, t2 in zip(times, times[1:])] if len(times) > 1 else [0]
        records.append({
            'src': src,
            'dst': dst,
            'proto': proto,
            'sport': sport or '',
            'dport': dport or '',
            'bytes': v['bytes'],
            'pkts': v['pkts'],
            'duration': times[-1] - times[0],
            'iat_mean': sum(iat) / len(iat),
            'iat_std': pd.Series(iat).std()
        })

    return pd.DataFrame(records)

if __name__ == '__main__':
    # Processa normal e attack, rotula e salva
    df_norm = pcap_to_flows('data/raw/normal.pcap')
    df_norm['label'] = 0
    df_att  = pcap_to_flows('data/raw/attack.pcap')
    df_att['label'] = 1

    df_all = pd.concat([df_norm, df_att], ignore_index=True)
    df_all.to_csv('data/processed/flows.csv', index=False)
    print(f'Features extraídas: {df_all.shape[0]} fluxos → data/processed/flows.csv')

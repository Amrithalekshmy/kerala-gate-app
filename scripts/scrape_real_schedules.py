"""
Scrape real train schedules from confirmtkt.com for Kerala trains.
Outputs data/train_schedules.csv with accurate station-by-station timings.
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
import re
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
STATIONS_FILE = os.path.join(DATA_DIR, 'kerala_stations.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'train_schedules.csv')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

# Real Kerala train numbers — covering all major routes
TRAIN_NUMBERS = [
    # ==========================================
    # LONG DISTANCE / EXPRESS (through Kerala)
    # ==========================================

    # Mangalore/Konkan → TVC (Coastal southbound)
    16605,  # Mangala Lakshadweep Express (MAQ-TVC)
    16345,  # Netravati Express (LTT-TVC)
    12617,  # Mangala Express (CSMT-ERS)
    16525,  # Kanniyakumari Express (SBC-CAPE)
    12625,  # Kerala Express (NDLS-TVC → actually TVC-NDLS)
    16859,  # Mangalore-Chennai Express (MAQ-MAS)
    12695,  # Trivandrum Rajdhani (NDLS-TVC)
    22113,  # Kochuveli Superfast
    12431,  # Rajdhani Express (TVC-NZM)
    16335,  # Nagercoil Express
    12081,  # Jan Shatabdi (ERS-TVC, via Kottayam)
    12623,  # Thiruvananthapuram Mail
    12511,  # Rapti Sagar Express
    16527,  # Guruvayur Express (YPR-GUV)
    12201,  # Garib Rath (CSMT-TVC)
    16347,  # Mangalore Express (via Kottayam)

    # Northbound equivalents
    16606,  # Mangala Lakshadweep Express (TVC-MAQ)
    16346,  # Netravati Express (TVC-LTT)
    12618,  # Mangala Express (ERS-CSMT)
    16526,  # Kanniyakumari Express (CAPE-SBC)
    12626,  # Kerala Express (TVC-NDLS)
    16860,  # Chennai-Mangalore Express (MAS-MAQ)
    12696,  # Trivandrum Rajdhani (TVC-NDLS)
    22114,  # Kochuveli Superfast
    12432,  # Rajdhani Express (NZM-TVC)
    16336,  # Nagercoil Express
    12082,  # Jan Shatabdi (TVC-ERS)
    12624,  # Thiruvananthapuram Mail
    12512,  # Rapti Sagar Express
    16528,  # Guruvayur Express (GUV-YPR)
    12202,  # Garib Rath (TVC-CSMT)
    16348,  # Mangalore Express (via Kottayam)

    # ==========================================
    # INTERCITY / SHORT EXPRESS
    # ==========================================

    # Venad Express pair (SRR-TVC via Kottayam, very popular)
    16301,  # Venad Express (SRR-TVC)
    16302,  # Venad Express (TVC-SRR)

    # Vanchinad Express (SRR-TVC via Kottayam)
    16303,  # Vanchinad Express (SRR-TVC)
    16304,  # Vanchinad Express (TVC-SRR)

    # Ernad Express
    16305,  # Ernad Express (ERS-CAN)
    16306,  # Ernad Express (CAN-ERS)

    # Maveli Express (TVC-MAQ)
    16603,  # Maveli Express (TVC-MAQ)
    16604,  # Maveli Express (MAQ-TVC)

    # Parasuram Express (MAQ-TVC)
    16649,  # Parasuram Express (MAQ-TVC)
    16650,  # Parasuram Express (TVC-MAQ)

    # Malabar Express (TVC-MAQ)
    16629,  # Malabar Express (TVC-MAQ)
    16630,  # Malabar Express (MAQ-TVC)

    # Jan Shatabdi (TVC-CLT)
    12075,  # Jan Shatabdi (TVC-CLT)
    12076,  # Jan Shatabdi (CLT-TVC)

    # ERS-TVC Jan Shatabdi
    12077,  # Jan Shatabdi (ERS-TVC)
    12078,  # Jan Shatabdi (TVC-ERS)

    # Ernakulam-Kannur Intercity
    12083,  # ERS-CAN Express
    12084,  # CAN-ERS Express

    # Kanyakumari / Nagercoil area
    16723,  # Ananthapuri Express (MAQ-TVC)
    16724,  # Ananthapuri Express (TVC-MAQ)

    # ==========================================
    # PASSENGER / MEMU / DMU (frequent, short)
    # ==========================================

    # ERS-TVC Passenger/MEMU
    56361,  # Passenger (ERS-TVC)
    56362,  # Passenger (TVC-ERS)
    56371,  # Passenger (ERS-QLN)
    56372,  # Passenger (QLN-ERS)

    # CLT-ERS Passenger
    56381,  # Passenger (CLT-ERS)
    56382,  # Passenger (ERS-CLT)

    # CAN-CLT Passenger
    56365,  # Passenger (CAN-CLT)
    56366,  # Passenger (CLT-CAN)

    # SRR-PGT (Shoranur-Palakkad)
    56651,  # Passenger (SRR-PGT)
    56652,  # Passenger (PGT-SRR)
    56653,  # Passenger (SRR-PGT)
    56654,  # Passenger (PGT-SRR)

    # SRR-NIL (Shoranur-Nilambur)
    56601,  # Passenger (SRR-NIL)
    56602,  # Passenger (NIL-SRR)
    56603,  # Passenger (SRR-NIL)
    56604,  # Passenger (NIL-SRR)

    # TCR-GUV (Thrissur-Guruvayur)
    56605,  # Passenger (TCR-GUV)
    56606,  # Passenger (GUV-TCR)

    # MEMU trains
    66301,  # MEMU (ERS-CLT)
    66302,  # MEMU (CLT-ERS)
    66303,  # MEMU (ERS-TVC)
    66304,  # MEMU (TVC-ERS)
    66305,  # MEMU (ERS-CAN)
    66306,  # MEMU (CAN-ERS)
    66309,  # MEMU (ERS-SRR)
    66310,  # MEMU (SRR-ERS)
    66311,  # MEMU (TVC-QLN)
    66312,  # MEMU (QLN-TVC)

    # KYJ-PUU (Kayamkulam-Punalur)
    56393,  # Passenger (KYJ-PUU)
    56394,  # Passenger (PUU-KYJ)

    # QLN-TVC Passenger
    56375,  # Passenger (QLN-TVC)
    56376,  # Passenger (TVC-QLN)

    # TVC-PASA (south)
    56377,  # Passenger (TVC-NCJ)
    56378,  # Passenger (NCJ-TVC)

    # Kottayam line passengers
    56385,  # Passenger (ERS-KTYM)
    56386,  # Passenger (KTYM-ERS)

    # ==========================================
    # ADDITIONAL TRAINS FOR COVERAGE
    # ==========================================
    20923,  # Superfast (CBE-TVC)
    20924,  # Superfast (TVC-CBE)
    16341,  # Trivandrum Express (SRR-TVC)
    16342,  # Trivandrum Express (TVC-SRR)
    16307,  # Alleppey Express (SRR-ALLP)
    16308,  # Alleppey Express (ALLP-SRR)
    16343,  # Trivandrum Express
    16344,  # Trivandrum Express
    16349,  # Mangalore Express
    16350,  # Mangalore Express
    16329,  # Udyogamandal Express
    16330,  # Udyogamandal Express
    22633,  # Superfast (TVC-MAQ)
    22634,  # Superfast (MAQ-TVC)
    16609,  # Talghat Express (MAQ-TVC)
    16610,  # Talghat Express (TVC-MAQ)
]

# Remove duplicates and sort
TRAIN_NUMBERS = sorted(set(TRAIN_NUMBERS))


def load_kerala_stations():
    stations = set()
    with open(STATIONS_FILE, 'r') as f:
        for row in csv.DictReader(f):
            stations.add(row['station_code'].strip())
    return stations


def scrape_schedule(train_number):
    """Scrape schedule for one train from confirmtkt.com."""
    url = f'https://www.confirmtkt.com/train-schedule/{train_number}'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
    except requests.RequestException as e:
        print(f"  Error: {e}")
        return None

    soup = BeautifulSoup(resp.text, 'html.parser')
    tables = soup.find_all('table')
    if not tables:
        return None

    # Parse schedule table (first table)
    schedule = []
    header_row = tables[0].find('tr')
    if not header_row:
        return None

    rows = tables[0].find_all('tr')[1:]  # Skip header
    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all('td')]
        if len(cols) < 6:
            continue

        # Station format: 'Shoranur Jn - SRR'
        stn_text = cols[1]
        match = re.search(r'-\s*([A-Z]{2,6})\s*$', stn_text)
        code = match.group(1) if match else ''
        name = stn_text.rsplit('-', 1)[0].strip() if match else stn_text

        if not code:
            continue

        arr = cols[2].strip()
        dep = cols[3].strip()
        dist = cols[5].replace('km', '').strip()

        schedule.append({
            'station_code': code,
            'station_name': name,
            'arrival_time': '--' if arr in ('Start', 'Source', '-', '') else arr,
            'departure_time': '--' if dep in ('End', 'Destination', '-', '') else dep,
            'distance_km': dist,
        })

    if not schedule:
        return None

    # Get train name from title
    train_name = ''
    title = soup.find('title')
    if title:
        match = re.search(r'(\d+)\s+(.+?)\s*(?:Route|Schedule|Train)', title.get_text())
        if match:
            train_name = match.group(2).strip()
    if not train_name:
        h1 = soup.find('h1')
        if h1:
            match = re.search(r'\d+\s+(.+?)(?:\s+Route|\s+Schedule|$)', h1.get_text())
            if match:
                train_name = match.group(1).strip()

    # Get runs_on_days from info table
    runs_on = 'MTWTFSS'
    if len(tables) > 1:
        for row in tables[1].find_all('tr'):
            cols = [td.get_text(strip=True) for td in row.find_all('td')]
            if len(cols) >= 2 and 'Service' in cols[0]:
                days_text = cols[1]
                day_names = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3,
                            'Fri': 4, 'Sat': 5, 'Sun': 6}
                day_chars = list('-------')
                for day_name, idx in day_names.items():
                    if day_name in days_text:
                        day_chars[idx] = 'MTWTFSS'[idx]
                runs_on = ''.join(day_chars)

    return {
        'train_number': str(train_number),
        'train_name': train_name,
        'runs_on_days': runs_on,
        'schedule': schedule,
    }


def main():
    kerala_stations = load_kerala_stations()
    print(f"Loaded {len(kerala_stations)} Kerala station codes")
    print(f"Scraping {len(TRAIN_NUMBERS)} trains from confirmtkt.com...")
    print()

    all_rows = []
    scraped = 0
    failed = 0
    skipped = 0

    for i, train_num in enumerate(TRAIN_NUMBERS):
        print(f"[{i+1}/{len(TRAIN_NUMBERS)}] Train {train_num}...", end=' ', flush=True)

        result = scrape_schedule(train_num)

        if result and result['schedule']:
            # Check Kerala station overlap
            train_codes = {s['station_code'] for s in result['schedule']}
            kerala_overlap = train_codes & kerala_stations

            if kerala_overlap:
                for stop in result['schedule']:
                    all_rows.append({
                        'train_number': result['train_number'],
                        'train_name': result['train_name'],
                        'station_code': stop['station_code'],
                        'station_name': stop['station_name'],
                        'arrival_time': stop['arrival_time'],
                        'departure_time': stop['departure_time'],
                        'distance_km': stop['distance_km'],
                        'day_of_journey': 1,
                        'runs_on_days': result['runs_on_days'],
                    })
                print(f"{result['train_name']} — {len(result['schedule'])} stops, "
                      f"{len(kerala_overlap)} in Kerala")
                scraped += 1
            else:
                print("No Kerala stations — skipped")
                skipped += 1
        else:
            print("NOT FOUND")
            failed += 1

        # Be polite — 1.5s delay between requests
        time.sleep(1.5)

    # Write output CSV
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'train_number', 'train_name', 'station_code', 'station_name',
            'arrival_time', 'departure_time', 'distance_km',
            'day_of_journey', 'runs_on_days'
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'='*50}")
    print(f"Done! Scraped: {scraped}, Failed: {failed}, Skipped: {skipped}")
    print(f"Total schedule rows: {len(all_rows)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()

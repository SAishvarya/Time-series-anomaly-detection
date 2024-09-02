import json
import random
import time
from openpyxl import Workbook
from datetime import datetime, timezone, timedelta
import copy

# Path to the feature flag JSON file inside the container
FEATURE_FLAG_FILE = '/app/demo.flagd.json'
EXCEL_FILE = '/app/anomaly_log.xlsx'

Feature_Flag = [
    'adServiceFailure', 'adServiceManualGc', 'adServiceHighCpu', 'cartServiceFailure', 'productCatalogFailure',
    'recommendationServiceCacheFailure', 'paymentServiceFailure', 'paymentServiceUnreachable', 'loadgeneratorFloodHomepage', 'kafkaQueueProblems', 'imageSlowLoad'
]

ANOMALY_PROBABILITY = 0.2

def load_feature_flags():
    print("Loading feature flags from JSON file...")
    with open(FEATURE_FLAG_FILE, 'r') as file:
        return json.load(file)

def reset_feature_flags(feature_flags):
    print("Resetting feature flags to False...")
    # Initialize all feature flags to False (off)
    flags = copy.deepcopy(feature_flags)
    for ff in Feature_Flag:
        flags[ff] = False
    return flags

def toggle_feature_flag(flags, service, value):
    print(f"Toggling feature flag {service} to {value}...")

    flags[service] = value
def log_anomaly_to_excel(workbook, start_time, end_time, anomaly_type, flag):
    sheet = workbook.active
    sheet.append([start_time.isoformat(), end_time.isoformat(), anomaly_type, flag])
    print(f"Logged anomaly: {start_time}, {end_time}, {anomaly_type}, {flag}")

def random_scenario():
    feature_flags = load_feature_flags()
    flags = reset_feature_flags(feature_flags)

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(['Start Time (UTC)', 'End Time (UTC)', 'Anomaly Type', 'Flag(s)'])

    for _ in iter(int, 1):
        if random.random() < ANOMALY_PROBABILITY:
            # Choose an anomaly scenario
            scenario_type = random.choice(['point', 's_long', 'multiple'])
        else:
            # Choose a normal scenario
            scenario_type = 'normal'

        if scenario_type == 'normal':
            # In normal scenario, all feature flags should be off
            print("Normal scenario")
            for service in flags:
                toggle_feature_flag(flags, service, False)

            time.sleep(random.uniform(1, 5))

        elif scenario_type == 'point':
            # Perform a point anomaly scenario
            ff = random.choice(Feature_Flag)
            start_time = datetime.now(timezone.utc)
            toggle_feature_flag(flags, ff, True)
            time.sleep(random.uniform(1, 5))
            end_time = datetime.now(timezone.utc)
            toggle_feature_flag(flags, ff, False)
            log_anomaly_to_excel(workbook, start_time, end_time, 'point', ff)

        elif scenario_type == 's_long':
            # Perform a long scenario anomaly
            ff = random.choice(Feature_Flag)
            start_time = datetime.now(timezone.utc)
            toggle_feature_flag(flags, ff, True)
            time.sleep(random.uniform(5, 10))
            end_time = datetime.now(timezone.utc)
            toggle_feature_flag(flags, ff, False)
            log_anomaly_to_excel(workbook, start_time, end_time, 's_long', ff)

        elif scenario_type == 'multiple':
            # Perform a multiple anomaly scenario
            service_count = random.randint(2, len(Feature_Flag))
            ff = random.sample(Feature_Flag, service_count)
            start_time = datetime.now(timezone.utc)
            for service in ff:
                toggle_feature_flag(flags, service, True)
            time.sleep(random.uniform(1, 10))
            end_time = datetime.now(timezone.utc)
            for service in ff:
                toggle_feature_flag(flags, service, False)
            log_anomaly_to_excel(workbook, start_time, end_time, 'multiple', ','.join(ff))

    workbook.save(EXCEL_FILE)
    print(f"Workbook saved as {EXCEL_FILE}")

random_scenario()

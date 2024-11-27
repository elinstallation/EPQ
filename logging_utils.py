def log_prediction(gender, age):
    with open("recognition_log.txt", "a") as log_file:
        log_file.write(f"Gender: {gender}, Age: {age}\n")

def log_with_timestamp(gender, age):
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("recognition_log.txt", "a") as log_file:
        log_file.write(f"{timestamp}: Gender: {gender}, Age: {age}\n")

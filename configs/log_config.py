
import logging
import os
import re
import datetime

LOGGER_NAME = "langtime"

def manage_log_files(directory, latest_file=10):
    def find_log_files(directory):
        pattern = re.compile(r"run_(\d{8})_(\d{6})\.log")
        log_files = []

        for filename in os.listdir(directory):
            match = pattern.match(filename)
            if match:
                date_part = match.group(1)
                time_part = match.group(2)
                dt_str = f"{date_part} {time_part}"
                dt = datetime.strptime(dt_str, "%Y%m%d %H%M%S")
                log_files.append((dt, filename))

        return log_files

    log_files = find_log_files(directory)
    log_files.sort(reverse=True) # Sort by datetime in descending order

    # Keep the latest files
    to_keep = log_files[:latest_file]
    to_delete = log_files[latest_file:]

    # print(f"Keeping {latest_file} files:")
    # for _, filename in to_keep:
    #     print(f"  - {filename}")

    # print(f"\nDeleting {len(to_delete)} files:")
    # for _, filename in to_delete:
    #     file_path = os.path.join(directory, filename)
    #     os.remove(file_path)
    #     print(f"  - {filename}")


def setup_logging(log_file: str, logger_name="time_qwen", log_level=logging.DEBUG, local_rank=-1):
    global LOGGER_NAME
    LOGGER_NAME = logger_name
    if log_file == "fixed":
        file_name = "logs/run.log"
        if os.path.exists(file_name):
            os.remove(file_name)
    elif log_file == "time":
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        # 生成文件名
        file_name = f"logs/run_{timestamp}.log"
    elif log_file.endswith(".log"):
        file_name = log_file
    else:
        raise ValueError("Invalid log file name format")

    logger = logging.getLogger(LOGGER_NAME)


    fh = logging.FileHandler(file_name)
    
    ch = logging.StreamHandler()

    if local_rank < 1:
        logger.setLevel(log_level)
        fh.setLevel(log_level)
        ch.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.CRITICAL)
        fh.setLevel(logging.CRITICAL)
        ch.setLevel(logging.CRITICAL)
    
    formatter = logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def get_logger():
    return logging.getLogger(LOGGER_NAME)

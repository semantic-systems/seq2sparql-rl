import re
import json

ENTITY_PATTERN = re.compile('Q[0-9]+')
PREDICATE_PATTERN = re.compile('P[0-9]+')

def is_timestamp(timestamp):
    pattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    if not(pattern.match(timestamp)):
        return False
    else:
        return True


def convertTimestamp(timestamp):
    yearPattern = re.compile('^[0-9][0-9][0-9][0-9]-00-00T00:00:00Z')
    monthPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-00T00:00:00Z')
    dayPattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    timesplits = timestamp.split("-")
    year = timesplits[0]
    if yearPattern.match(timestamp):
        return year
    month = convertMonth(timesplits[1])
    if monthPattern.match(timestamp):
        return month + " " + year
    elif dayPattern.match(timestamp):
        day = timesplits[2].rsplit("T")[0]
        return day + " " + month + " " + year

    return timestamp

def convertMonth(month):
    return{
        "01": "january",
        "02": "february",
        "03": "march",
        "04": "april",
        "05": "may",
        "06": "june",
        "07": "july",
        "08": "august",
        "09": "september",
        "10": "october",
        "11": "november",
        "12": "december"
    }[month]


#!/bin/bash

sqlite3 $1 "select stat_pid, stat_ppid, avg(stat_rss / 1024.0 * 4), max(stat_rss / 1024.0 * 4) from record group by stat_pid"


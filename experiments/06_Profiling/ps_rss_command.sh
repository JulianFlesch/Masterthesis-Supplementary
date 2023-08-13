#!/bin/bash

ps -o rss= -p $1 | awk '{print $1*4 "KiB"}'

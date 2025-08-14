#!/usr/bin/env bash

YELLOW='\033[0;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
SKYBLUE='\033[0;36m'
PLAIN='\033[0m'

speed_test(){
    local id=$1
    local name=$2
    local log=$(uvx speedtest-cli --server "$id" --simple 2>/dev/null)
    local ping=$(echo "$log" | awk '/Ping/{print $2}')
    local download=$(echo "$log" | awk '/Download/{print $2}')
    local upload=$(echo "$log" | awk '/Upload/{print $2}')
    echo -e "${YELLOW}$name${PLAIN}  ↑ ${GREEN}${upload} Mbit/s${PLAIN}  ↓ ${RED}${download} Mbit/s${PLAIN}  延迟 ${SKYBLUE}${ping} ms${PLAIN}"
}

print_china_speedtest(){
    echo "=== 中国节点测速 ==="
    speed_test '36663' '镇江 5G CT'
    speed_test '26352' '南京 5G CT'
    speed_test '59386' '杭州 CT'
    speed_test '5145'  '北京 CU'
    speed_test '24447' '上海 5G CU'
    speed_test '27154' '天津 5G CU'
    speed_test '25637' '上海 5G CM'
    speed_test '26940' '银川 CM'
    speed_test '27249' '南京 5G CM'
    speed_test '25858' '北京 CM'
    speed_test '4575'  '成都 CM'
}

print_global_speedtest(){
    echo "=== 全球节点测速 ==="
    speed_test '1536'  '香港'
    speed_test '18611' '台湾'
    speed_test '40508' '新加坡'
    speed_test '56935' '东京'
    speed_test '67514' '首尔'
    speed_test '18229' '洛杉矶'
    speed_test '24281' '伦敦'
    speed_test '53651' '法兰克福'
    speed_test '21268' '法国'
}

case "$1" in
    china)
        print_china_speedtest
        ;;
    global)
        print_global_speedtest
        ;;
    all|"")
        print_china_speedtest
        print_global_speedtest
        ;;
esac
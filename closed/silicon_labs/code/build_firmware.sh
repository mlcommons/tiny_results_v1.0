#!/bin/bash

# build_firmware.sh
# Generates firmware binaries for all benchmarks for the EFR32xG24 Dev Kit
# (brd2601b). By default generates FW for all models (vww, kws, ic, ad)
# and modes (energy, performance), using a baud rate of 115200 for the
# performance mode.

# Exit early whenever a sub process fails
set -e

usage() {
echo "$(basename "$0"): $build_firmware.sh [-fh] [-s baudrate] [-b bencmhark] [-m mode]"
echo "  -f      		enable fast regenerate/rebuild (do not clean build dir)"
echo "  -h 			 	display help"
echo "  -s baudrate     specify VCOM baud rate to use for performance firmware [115200 | 921600]"
echo "  -b benchmark    specify to generate firmware for single model: [vww | ad | ic | kws]"
echo "  -m mode         specify to generate firmware for single mode: [performance | mode]"
exit
}

while getopts s:b:m:fh  flag
do
	case "${flag}" in
		s) BAUDRATE=${OPTARG};;
		b) BENCHMARK=${OPTARG};;
		m) MODES=${OPTARG};;
		f) FAST=true;;
		h) usage
	esac
done

# Set default baudrate to 115200
if [ -z ${BAUDRATE} ]
then
	BAUDRATE="115200"
fi

# Set to generate for all benchmarks as default
if [ -z ${BENCHMARK} ]
then
	BENCHMARK="vww ic kws ad"
fi

if [ -z ${MODES} ]
then
  MODES="performance energy"
fi

# Clean build directory
if [ "${FAST}" = "true" ]
then
  echo "Fast mode enabled, not removing build directory"
else
  echo "Removing build directory"
  rm -rf build
  mkdir -p build/energy
  mkdir -p build/performance
fi

# Generate firmware binaries for brd2601b
declare -A BM_MAP=(["kws"]="keyword_spotting" ["ad"]="anomaly_detection" ["ic"]="image_classification" ["vww"]="person_detection")

for BM in ${BENCHMARK}
do
	for MODE in ${MODES}
	do
    # Set correct configurations based on benchmark mode
		if [ ${MODE} = "energy" ]
		then
			CONFIG_OPTS="EE_CFG_ENERGY_MODE:0,SL_IOSTREAM_EUSART_VCOM_BAUDRATE:${BAUDRATE}"
		else
      CONFIG_OPTS="EE_CFG_ENERGY_MODE:1,SL_IOSTREAM_EUSART_VCOM_BAUDRATE:${BAUDRATE}"
    fi

    	slc generate -p ${BM_MAP[${BM}]}/mlperf_tiny_${BM}.slcp -d "build/${MODE}/${BM}" --with "brd2601b" --configuration "${CONFIG_OPTS}"
		make -f mlperf_tiny_${BM}.Makefile -C "build/${MODE}/${BM}" -j release
		cp -v build/${MODE}/${BM}/build/release/mlperf_tiny_${BM}.s37 build/${MODE}/
	done
done

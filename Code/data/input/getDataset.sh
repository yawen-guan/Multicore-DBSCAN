#!/bin/bash

mkdir tmp && cd tmp
wget http://yikeqiaomu.top/nextcloud/index.php/s/gR3YWfGKRiZpPXF/download/gaia_dr2_ra_dec_50M.tar.gz
tar zxvf gaia_dr2_ra_dec_50M.tar.gz
mv gaia_dr2_ra_dec_50M.dat ../ && cd ..
rm -rf tmp/
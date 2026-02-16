#!/bin/bash

bash -c "pkill -9 python3"
bash -c "pkill -9 python"
bash -c "pkill -9 pt_main_thread"
bash -c "pkill -9 pt_data_worker"

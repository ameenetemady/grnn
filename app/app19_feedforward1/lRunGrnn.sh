#!/bin/bash

th ../common/run_grnn.lua true 0.3
th ../common/run_grnn.lua true 0.2
th ../common/run_grnn.lua true 0.15
th ../common/run_grnn.lua true 0.1
th ../common/run_grnn.lua true 0.05
th ../common/run_grnn.lua true 0.02
th ../common/run_grnn.lua true 0.01
th ../common/run_grnn.lua true 0.005
th ../common/run_grnn.lua true 0.002
th ../common/run_grnn.lua true 0.001
th ../common/run_grnn.lua true 0.000

th ../common/run_grnn.lua false 0.3
th ../common/run_grnn.lua false 0.2
th ../common/run_grnn.lua false 0.15
th ../common/run_grnn.lua false 0.1
th ../common/run_grnn.lua false 0.05
th ../common/run_grnn.lua false 0.02
th ../common/run_grnn.lua false 0.01
th ../common/run_grnn.lua false 0.005
th ../common/run_grnn.lua false 0.002
th ../common/run_grnn.lua false 0.001
th ../common/run_grnn.lua false 0.000


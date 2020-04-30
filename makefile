
ORIGINAL_ZIP=original/Benchmark_Data.zip

ORIGINAL_DIR=data/00-original
ORIGINAL_STATUS=$(ORIGINAL_DIR)/status

FLAT_DIR=data/01-flat
FLAT_STATUS=$(FLAT_DIR)/status
FLAT_GENERATOR=00-original2flat.py
FLAT_DATA_A=$(FLAT_DIR)/Supp-A-36630-HPRD-positive-interaction.txt
FLAT_DATA_B=$(FLAT_DIR)/Supp-B-36480-HPRD-negative-interaction.txt
FLAT_DATA_C=$(FLAT_DIR)/Supp-C-3899-HPRD-positive-interaction-below-0.25.txt
FLAT_DATA_D=$(FLAT_DIR)/Supp-D-4262-HPRD-negative-interaction-below-0.25.txt
FLAT_DATA_E=$(FLAT_DIR)/Supp-E-1882-interacting-0.5-non-interacting-0.5.txt

CT_DIR=data/02-ct
CT_STATUS=$(CT_DIR)/status
CT_GENERATOR=01-flat2ct.py

CT_BIN_DIR=data/02-ct-bin
CT_BIN_STATUS=$(CT_BIN_DIR)/status
CT_BIN_GENERATOR=01-flat2ct-bin.py

AC_DIR=data/03-ac
AC_STATUS=$(AC_DIR)/status
AC_GENERATOR=02-flat2ac.py

AC_BIN_DIR=data/03-ac-bin
AC_BIN_STATUS=$(AC_BIN_DIR)/status
AC_BIN_GENERATOR=02-flat2ac-bin.py


CD_CT_ZIP=data/Benchmark_CD_CT.zip
CD_AC_ZIP=data/Benchmark_CD_AC.zip


MODEL_RESULTS=$(CT_MODEL_RESULTS) $(AC_MODEL_RESULTS) $(CD_RESULTS) $(PREDICT_PPI_RESULTS)
FINAL_RESULTS=$(subst model/,results/, $(MODEL_RESULTS))

.PHONY: default
default: main

$(ORIGINAL_STATUS):$(ORIGINAL_ZIP)
	mkdir -p $(ORIGINAL_DIR)
	unzip -o $(ORIGINAL_ZIP) -d $(ORIGINAL_DIR)
	@touch -m $(ORIGINAL_STATUS)
	@echo "$(ORIGINAL_DIR) is ok"

$(FLAT_STATUS):$(FLAT_GENERATOR) $(ORIGINAL_STATUS)
	python $(FLAT_GENERATOR)
	@touch -m $(FLAT_STATUS)
	@echo "$(FLAT_DIR) is ok"

.PHONY:statistics
statistics: statistics.py $(FLAT_STATUS)
	python statistics.py --input $(FLAT_DATA_A) --output $(FLAT_DATA_A).csv
	python statistics.py --input $(FLAT_DATA_B) --output $(FLAT_DATA_B).csv
	python statistics.py --input $(FLAT_DATA_C) --output $(FLAT_DATA_C).csv
	python statistics.py --input $(FLAT_DATA_D) --output $(FLAT_DATA_D).csv
	python statistics.py --input $(FLAT_DATA_E) --output $(FLAT_DATA_E).csv

$(CT_STATUS):$(CT_GENERATOR) $(FLAT_STATUS)
	python $(CT_GENERATOR)
	@touch -m $(CT_STATUS)
	@echo "$(CT_DIR) is ok"

$(CT_BIN_STATUS):$(CT_BIN_GENERATOR) $(CT_STATUS)
	python $(CT_BIN_GENERATOR)
	@touch -m $(CT_BIN_STATUS)
	@echo "$(CT_BIN_DIR) is ok"

$(AC_STATUS):$(AC_GENERATOR) $(FLAT_STATUS)
	python $(AC_GENERATOR)
	@touch -m $(AC_STATUS)
	@echo "$(AC_DIR) is ok"

$(AC_BIN_STATUS):$(AC_BIN_GENERATOR) $(AC_STATUS)
	python $(AC_BIN_GENERATOR)
	@touch -m $(AC_BIN_STATUS)
	@echo "$(AC_BIN_DIR) is ok"


$(CD_CT_ZIP):$(FLAT_DATA_C) $(FLAT_DATA_D)
	python util_create_ppi_dataset.py -p "$(FLAT_DATA_C)" -n "$(FLAT_DATA_D)" -o "$(patsubst %.zip,%, $(CD_CT_ZIP))" -m ct -s 0.6
$(CD_AC_ZIP):$(FLAT_DATA_C) $(FLAT_DATA_D)
	python util_create_ppi_dataset.py -p "$(FLAT_DATA_C)" -n "$(FLAT_DATA_D)" -o "$(patsubst %.zip,%, $(CD_AC_ZIP))" -m ac -s 0.6

$(CD_DATAS_STATUS):$(CD_CT_ZIP) $(CD_AC_ZIP)
	unzip -o $(CD_CT_ZIP) -d $(patsubst %.zip,%, $(CD_CT_ZIP))
	unzip -o $(CD_AC_ZIP) -d $(patsubst %.zip,%, $(CD_AC_ZIP))
	@touch -m $(CD_DATAS_STATUS)

.PHONY:data
data: $(CT_BIN_STATUS) $(AC_BIN_STATUS)	$(CD_DATAS_STATUS)

.PHONY:main
main: data

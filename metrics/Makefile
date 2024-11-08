DIRS := reference starting_kit       # List the folders you want to zip
PROGRAM_FILES := evaluate.py metadata
PROGRAM_ZIP := program.zip
TOP_ZIP := wriva-codabench-competition.zip      # Name of the top-level zip file
DIR_ZIPS := $(DIRS:%=%.zip)   # Create a list of zip files for the directories

# Default target to zip directories and all files
all: $(TOP_ZIP)

# Rule to zip each folder into its own zip file
%.zip: %
	@echo "Zipping directory $<..."
	zip -r $@ $<

$(PROGRAM_ZIP): $(PROGRAM_FILES)
	@echo "Zipping root files..."
	zip -r $(PROGRAM_ZIP) $(PROGRAM_FILES)

# Rule to zip everything in the directory, including the created zip files
$(TOP_ZIP): $(DIR_ZIPS) $(PROGRAM_ZIP)
	@echo "Zipping all files into $(TOP_ZIP)..."
	zip -r $(TOP_ZIP) *.html *.py *.jpg metadata *.yaml $(DIR_ZIPS) $(PROGRAM_ZIP)

# Clean up all zip files
clean:
	@echo "Cleaning up zip files..."
	rm -f $(DIR_ZIPS) $(TOP_ZIP) $(PROGRAM_ZIP)

.PHONY: all clean


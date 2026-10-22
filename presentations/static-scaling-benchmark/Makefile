# Makefile for Static Scaling Benchmark Typst Presentation

TYPST ?= typst
MAIN = main.typ
OUTPUT = presentation.pdf

.PHONY: all pdf png clean watch

all: pdf

pdf: $(OUTPUT)

$(OUTPUT): $(MAIN) components/*.typ slides/*.typ
	$(TYPST) compile $(MAIN) $(OUTPUT)
	@echo "Built $(OUTPUT) successfully."

png: $(MAIN) components/*.typ slides/*.typ
	rm -rf dist/slides
	mkdir -p dist/slides
	$(TYPST) compile --format png $(MAIN) dist/slides/slide-{n}.png
	@echo "Generated slide images in dist/slides/"

watch:
	$(TYPST) watch $(MAIN) $(OUTPUT)

clean:
	rm -f $(OUTPUT)
	rm -rf dist/

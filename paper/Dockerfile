FROM debian:bookworm-slim AS build

ENV DEBIAN_FRONTEND=noninteractive \
    SOURCE_DATE_EPOCH=1784300400

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        latexmk \
        texlive-fonts-recommended \
        texlive-lang-japanese \
        texlive-latex-extra \
        texlive-luatex \
        texlive-pictures \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work/paper

COPY paper/paper.tex ./paper.tex
COPY paper/paper-ja.tex ./paper-ja.tex
COPY workspace/plot /work/workspace/plot

RUN latexmk -lualatex -interaction=nonstopmode -halt-on-error paper.tex \
    && latexmk -lualatex -interaction=nonstopmode -halt-on-error paper-ja.tex \
    && test -s paper.pdf \
    && test -s paper-ja.pdf

FROM scratch AS artifact
COPY --from=build /work/paper/paper.pdf /paper.pdf
COPY --from=build /work/paper/paper-ja.pdf /paper-ja.pdf

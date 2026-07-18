# BPH-GPU papers

The English paper and its equivalent Japanese review copy are built with
LuaLaTeX inside Docker. From the repository root, run:

```sh
just --justfile paper/justfile
```

The equivalent direct command is:

```sh
docker build -f paper/Dockerfile --target artifact --output type=tar,dest=- . \
  | tar --extract --directory=paper --no-same-owner
```

Both commands produce `paper/paper.pdf` and `paper/paper-ja.pdf`. The Docker
build only typesets the papers; it uses the checked-in plots and does not rerun
the GPU experiments.

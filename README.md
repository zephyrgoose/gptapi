## Cloning the Repository

This repository uses submodules!

The profiles used to call the gptapi are stored as their own repository: https://github.com/zephyrgoose/gptapi-profiles-public

To clone this repository with all its submodules, use the following command:

```bash
git clone --recurse-submodules https://github.com/zephyrgoose/gptapi.git
```

If you've already cloned the repository without `--recurse-submodules`, you can initialise and update the submodules with:
```bash
git submodule update --init --recursive
```

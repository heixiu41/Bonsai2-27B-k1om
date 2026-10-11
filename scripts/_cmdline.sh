for p in "$@"; do
  printf '%s=' "$p"
  tr '\0' ' ' < /proc/$p/cmdline
  echo
done

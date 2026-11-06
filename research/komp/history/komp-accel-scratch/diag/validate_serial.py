"""validate_block_komp.py with the one-pass serial scan forced for every shape (the default only uses it
when batch-heads x value tiles >= SCAN_SERIAL, which no acceptance shape reaches)."""
import sys
from komp_slot import block_fast as BF
import validate_block_komp

BF.SCAN_SERIAL = 1
sys.argv = ['validate_block_komp.py', '--impl', 'fast', '--output', sys.argv[1]]
validate_block_komp.main()

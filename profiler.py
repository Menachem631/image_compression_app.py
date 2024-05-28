import cProfile
import pstats

import main

profiler = cProfile.Profile()
profiler.enable()
main_new.do_everything()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('tottime')
stats.print_stats()

#include <hwloc.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s out.xml [first last]\n", argv[0]); return 2; }
    unsigned first = argc > 2 ? (unsigned)atoi(argv[2]) : 0;
    unsigned last  = argc > 3 ? (unsigned)atoi(argv[3]) : 3;

    hwloc_topology_t t;
    if (hwloc_topology_init(&t) != 0) { fprintf(stderr, "init failed\n"); return 1; }
    if (hwloc_topology_load(t) != 0) { fprintf(stderr, "load failed\n"); return 1; }

    hwloc_bitmap_t set = hwloc_bitmap_alloc();
    hwloc_bitmap_set_range(set, first, last);
    if (hwloc_topology_restrict(t, set, 0) != 0) { fprintf(stderr, "restrict failed\n"); return 1; }

    if (hwloc_topology_export_xml(t, argv[1], 0) != 0) { fprintf(stderr, "export failed\n"); return 1; }

    printf("cores=%d pus=%d\n",
           hwloc_get_nbobjs_by_type(t, HWLOC_OBJ_CORE),
           hwloc_get_nbobjs_by_type(t, HWLOC_OBJ_PU));
    hwloc_bitmap_free(set);
    hwloc_topology_destroy(t);
    return 0;
}

import tsukuba as t
import grid as g
import exercise_3 as ex

graphs = []
models = t.all_models()
# Graph = g.determine_grid(nodes, edges)
for model in models:
    nodes, edges = model
    grid = g.determine_grid(nodes, edges)
    decomp = g.row_column_decomposition(grid)
    ex.min_sum_diffusion_accumulation(nodes, edges, grid, 1)
    #ex.min_sum_diffusion_method(nodes, edges, grid)

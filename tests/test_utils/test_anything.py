
#%%
colors = {"red" : 100, "blue" : 50, "purple" : 75}
# Delete the pair with a key of "red."
partial = ('re', 'blu', 'wh')
new_colors = {k:v for k,v in colors.items() if k.startswith(partial)}
print(new_colors)


#%%

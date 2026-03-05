import traceback
try:
    import app
    print('Imported app successfully')
except Exception as e:
    traceback.print_exc()
    raise

from src import get_model, config
config.parse_args()

model = get_model()
print(model.__class__.__name__)
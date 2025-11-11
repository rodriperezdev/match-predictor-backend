from model_training import train_pipeline

if __name__ == "__main__":
    print("Starting model training...")
    model, scaler, features, results = train_pipeline()
    print("\n[OK] Training complete!")
    print("\nModel Results:")
    for name, acc in results.items():
        print(f"  {name}: {acc:.4f}")
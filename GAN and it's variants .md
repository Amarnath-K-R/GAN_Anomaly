---

# GAN Variants for Anomaly Detection

This README provides an overview of three Generative Adversarial Network (GAN) architectures—**GAN**, **AnoGAN**, and **f-AnoGAN**—focusing on their purpose, architecture, and suitability for anomaly detection tasks, especially in real-time scenarios like tsunami detection.

---

## 1. GAN (Generative Adversarial Network)
- **Purpose**: General-purpose framework for generating realistic data by learning a data distribution.
- **Architecture**: GANs consist of two networks—a generator and a discriminator—that compete with each other:
    - The **generator** creates synthetic data.
    - The **discriminator** tries to distinguish between real and synthetic data.
- **Usage**: Primarily used for data generation, such as generating images, videos, and other types of data, by capturing the underlying distribution of the training data.
- **Limitation**: GANs are not designed explicitly for anomaly detection. While they can model “normal” data distributions, they lack a structured way to identify and quantify anomalies.

---

## 2. AnoGAN (Anomaly Detection GAN)
- **Purpose**: Designed specifically for anomaly detection tasks, such as in medical imaging, industrial monitoring, or tsunami detection.
- **Architecture**: AnoGAN also includes a generator and a discriminator but adds a **latent search process** to evaluate anomalies.
- **Anomaly Detection Mechanism**:
    - **Latent Space Mapping**: AnoGAN learns the normal data distribution during training. During anomaly detection, it searches for a latent vector that enables the generator to produce an image matching a test input image.
    - **Anomaly Score**: Calculated based on:
        - **Reconstruction error** (difference between input and generated image).
        - **Discriminator score** (likelihood of being real data).
- **Limitations**: The latent search process can be computationally slow, as it requires iterative optimization for each test sample, which limits real-time applications.

---

## 3. f-AnoGAN (Fast AnoGAN)
- **Purpose**: A faster, more efficient variant of AnoGAN for real-time anomaly detection applications.
- **Architecture**: f-AnoGAN introduces a **feature matching network (encoder)** to overcome AnoGAN’s slow latent search process.
- **Anomaly Detection Mechanism**:
    - **Encoder Network**: Adds an encoder that directly maps input data to the generator’s latent space, replacing the time-consuming optimization step in AnoGAN and enabling faster anomaly scoring.
    - **Anomaly Score**: Uses a combination of:
        - **Reconstruction error**.
        - **Discriminator output**.
- **Advantages**: f-AnoGAN achieves significantly faster inference times compared to AnoGAN, making it suitable for real-time or near-real-time anomaly detection tasks.

---

## Summary of Differences

| Feature               | **GAN**                     | **AnoGAN**                                   | **f-AnoGAN**                        |
|-----------------------|-----------------------------|----------------------------------------------|-------------------------------------|
| **Primary Use**       | Data generation             | Anomaly detection                            | Real-time anomaly detection         |
| **Architecture**      | Generator + Discriminator   | Generator + Discriminator + Latent Search    | Generator + Discriminator + Encoder |
| **Anomaly Score**     | Not directly available      | Reconstruction + Discriminator Score         | Reconstruction + Discriminator Score |
| **Latent Mapping**    | Not applicable              | Iterative optimization (slow)                | Encoder network (fast)              |
| **Speed**             | Fast for generation         | Slow due to latent search                    | Fast due to direct encoding         |
| **Real-time Suitability** | Limited                 | Limited                                      | High                                |

---

## Recommendations
- **AnoGAN**: Effective for tasks prioritizing anomaly detection accuracy over speed.
- **f-AnoGAN**: Recommended for real-time anomaly detection tasks, such as tsunami detection, where prompt alerts are essential, as it provides a faster solution while maintaining accuracy.

---

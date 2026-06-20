from pydantic import BaseModel, Field, ConfigDict


class CancerFeature(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float

    concave_points_mean: float = Field(
        alias="concave points_mean"
    )

    symmetry_mean: float

    radius_se: float
    perimeter_se: float
    area_se: float

    compactness_se: float
    concavity_se: float

    concave_points_se: float = Field(
        alias="concave points_se"
    )

    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float

    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float

    concave_points_worst: float = Field(
        alias="concave points_worst"
    )

    symmetry_worst: float
    fractal_dimension_worst: float


class Predict(BaseModel):
    diagnosis: str
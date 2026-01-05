checkpoint breakup_shape:
    message:
        "Break up {wildcards.shape} into the configured subunits."
    params:
        split_by=config["split_by"],
    input:
        script=workflow.source_path("../scripts/breakup_shape.py"),
        shapes="resources/user/shapes/{shape}.parquet",
    output:
        directory("resources/automatic/shapes/{shape}"),
    log:
        "logs/{shape}/breakup_shape.log",
    conda:
        "../envs/default.yaml"
    shell:
        """
        python {input.script:q} {input.shapes:q} {params.split_by:q} {output:q} 2> {log:q}
        """


rule prepare_resampled_inputs:
    message:
        "Resample inputs for {wildcards.subunit} in {wildcards.shape} to the projection and resolution of the land cover data, while aggregating land cover types."
    params:
        # Use internal defaults if not overridden
        land_cover_types_yaml_string=internal["land_cover_types"]
        | config.get("land_cover_types", {}),
    input:
        script=workflow.source_path("../scripts/resample.py"),
        shapes=rules.breakup_shape.output,
        land_cover_path=rules.clip_landcover.output,
        slope_path=rules.clip_slope.output,
        settlement_path=rules.clip_settlement.output,
        bathymetry_path=rules.clip_bathymetry.output,
        protected_area_path=rules.rasterise_clip_wdpa.output,
        solar_atlas_path=rules.clip_solar_atlas.output,
        wind_atlas_path=rules.clip_wind_atlas.output,
    output:
        resampled_input="resources/automatic/resampled_inputs/{shape}/{subunit}.nc",
        plot=report(
            "resources/automatic/resampled_inputs/{shape}/{subunit}.png",
            category="resampled_input",
        ),
    log:
        "logs/{shape}/{subunit}/prepare_resampled_inputs.log",
    conda:
        "../envs/default.yaml"
    shell:
        """
        python {input.script:q} \
        "{input.shapes}/{wildcards.subunit}.parquet" \
        {input.land_cover_path:q} {input.slope_path:q} {input.settlement_path:q} {input.bathymetry_path:q} {input.protected_area_path:q} \
        {input.solar_atlas_path:q} {input.wind_atlas_path:q} {params.land_cover_types_yaml_string:q} \
        {output.resampled_input:q} {output.plot:q} 2> {log:q}
        """


rule area_potential:
    message:
        "Compute area potential for the tech {wildcards.tech} and {wildcards.subunit} in {wildcards.shape}."
    params:
        config=lambda wildcards: config["techs"][f"{wildcards.tech}"],
        subunit_override_config=lambda wildcards: config.get("overrides", {})
        .get(wildcards.subunit, {})
        .get(wildcards.tech, {}),
        buffer_crs=lambda wildcards: config["buffer_crs"],
    input:
        script=workflow.source_path("../scripts/area_potential.py"),
        shapes=rules.breakup_shape.output,
        resampled_path=rules.prepare_resampled_inputs.output.resampled_input,
    output:
        area_potential="results/{shape}/{subunit}/area_potential_{tech}.tif",
        plot=report(
            "results/{shape}/{subunit}/area_potential_{tech}.png",
            category="area_potential",
        ),
    log:
        "logs/{shape}/{subunit}/area_potential_{tech}.log",
    conda:
        "../envs/default.yaml"
    shell:
        """
        python {input.script:q} "{input.shapes}/{wildcards.subunit}.parquet" {input.resampled_path:q} {params.config:q} {params.buffer_crs:q} {output.area_potential:q} {output.plot:q} --override_config={params.subunit_override_config:q} 2> {log:q}
        """


rule yearly_production_tech:
    message:
        """
        Calculate the yearly production potential of each pixel and document the area used for such production.
        Also store the average LCOE of potential production in each pixel for each tech.
        Applicable: all types of PV and wind.
        """
    params:
        density=lambda wildcards: config["techs"][f"{wildcards.tech}"]["density"],
        lifetime=lambda wildcards: config["techs"][f"{wildcards.tech}"]["lifetime"],
        costs=lambda wildcards: config["techs"][f"{wildcards.tech}"]["costs"],
    input:
        area_potentials_path="results/{shape}/{subunit}/area_potential_{tech}.tif",
        resampled_path="resources/automatic/resampled_inputs/{shape}/{subunit}.nc",
    output:
        production_tech="results/{shape}/{subunit}/yearly_production_lcoe_{tech}.nc",
    log:
        "logs/{shape}/{subunit}/yearly_production_{tech}.log",
    conda:
        "../envs/default.yaml"
    script:
        "../scripts/yearly_production_tech.py"


rule land_share:
    message:
        """
        Synthesise the potential production of all technologies for the given shape, then
        produce data for land curves and supply cost curves for combined given technologies.
        Land sharing only applies to onshore wind and open field PV.
        """
    params:
        land_share_type=config.get("land_share_type", None),
    input:
        inputs=lambda wildcards: expand(
            "results/{shape}/{subunit}/yearly_production_lcoe_{tech}.nc",
            shape=wildcards.shape,
            subunit=wildcards.subunit,
            tech=config["techs"].keys(),
        ),
        resampled_input="resources/automatic/resampled_inputs/{shape}/{subunit}.nc",
    output:
        curve_data="results/{shape}/{subunit}/curves_data_prep.nc"
    conda:
        "../envs/default.yaml"
    script:
        "../scripts/land_share.py"


rule plot_supply_and_land_curves:
    message:
        """
        Plot the supply cost curve and land use curve for the given shape and subunit.
        """
    input:
        synth_ds=rules.land_share.output,
    params:
        single_tech=config.get("land_cost_curve_tech", {}),
    output:
        supply_curve_path="results/{shape}/{subunit}/supply_curve_{shape}_{subunit}.png",
        land_curve_path="results/{shape}/{subunit}/land_curve_{shape}_{subunit}.png",
    conda:
        "../envs/default.yaml"
    script:
        "../scripts/plot_supply_and_land_curves.py"


rule aggregate_area_potential:
    message:
        "Aggregate area potential for the tech {wildcards.tech} in {wildcards.shape}."
    input:
        get_subunits,
    output:
        aggregated_area_potential="results/{shape}/area_potential_{tech}.tif",
    log:
        "logs/{shape}/aggregate_area_potential_{tech}.log",
    conda:
        "../envs/default.yaml"
    shell:
        """
        gdalwarp --config GDAL_CACHEMAX 3000 -wm 3000 -of GTiff -co COMPRESS=LZW {input} {output.aggregated_area_potential:q}
        """


rule plot_aggregated_area_potential:
    message:
        "Plot aggregated area potential for the tech {wildcards.tech} in {wildcards.shape}."
    input:
        rules.aggregate_area_potential.output.aggregated_area_potential,
    output:
        report(
            "results/{shape}/area_potential_{tech}.png", category="area_potential_plot"
        ),
    log:
        "logs/{shape}/plot_aggregated_area_potential_{tech}.log",
    conda:
        "../envs/default.yaml"
    script:
        "../scripts/tif_to_png.py"


rule area_potential_report:
    message:
        "Generate an overview report of the area potential for all techs in shapes {wildcards.shape}."
    input:
        shapes="resources/user/shapes/{shape}.parquet",
        area_potentials=expand(
            "results/{{shape}}/area_potential_{tech}.tif",
            tech=config["techs"].keys(),
        ),
        area_potential_plots=expand(
            "results/{{shape}}/area_potential_{tech}.png",
            tech=config["techs"].keys(),
        ),
    output:
        csv="results/{shape}/area_potential_report.csv",
        html=report(
            "results/{shape}/area_potential_report.html",
            category="area_potential_report_table",
        ),
    log:
        "logs/{shape}/area_potential_report.log",
    conda:
        "../envs/default.yaml"
    script:
        "../scripts/report.py"

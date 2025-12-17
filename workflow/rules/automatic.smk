"""Rules to used to download automatic resource files."""

if config.get("tiny_files", False):

    ##
    # Directly download clipped slope and bathymetry data
    ##

    rule clip_slope:
        message:
            "Download slope data covering the bounds of the input shapefile."
        params:
            cog_url=internal["resources"]["automatic"]["slope"],
        input:
            vector="resources/user/shapes/{shape}.parquet",
        output:
            path="resources/automatic/cutout/{shape}/slope.tif",
        log:
            "logs/{shape}/clip_slope.log",
        wrapper:
            "v7.2.0/geo/rasterio/clip-geotiff"

    rule clip_bathymetry:
        message:
            "Download bathymetry data covering the bounds of the input shapefile."
        params:
            cog_url=internal["resources"]["automatic"]["bathymetry"],
        input:
            vector="resources/user/shapes/{shape}.parquet",
        output:
            path="resources/automatic/cutout/{shape}/bathymetry.tif",
        log:
            "logs/{shape}/clip_bathymetry.log",
        wrapper:
            "v7.2.0/geo/rasterio/clip-geotiff"

else:

    ##
    # Download global slope and bathymetry data, then clip the files locally
    ##

    rule download_slope:
        message:
            "Download global slope data."
        params:
            url=internal["resources"]["automatic"]["slope"],
        output:
            path="resources/automatic/global/slope.tif",
        log:
            "logs/download_slope.log",
        conda:
            "../envs/shell.yaml"
        shell:
            """
            curl -sSLo {output:q} {params.url:q}
            """

    rule download_bathymetry:
        message:
            "Download global bathymetry data."
        params:
            url=internal["resources"]["automatic"]["bathymetry"],
        output:
            path="resources/automatic/global/bathymetry.tif",
        log:
            "logs/download_bathymetry.log",
        conda:
            "../envs/shell.yaml"
        shell:
            """
            curl -sSLo {output:q} {params.url:q}
            """

    rule clip_slope:
        message:
            "Cut slope data to the bounds of the input shapefile."
        input:
            script=workflow.source_path("../scripts/clip_raster.py"),
            shapes="resources/user/shapes/{shape}.parquet",
            slope=rules.download_slope.output,
        output:
            "resources/automatic/cutout/{shape}/slope.tif",
        log:
            "logs/{shape}/clip_slope.log",
        conda:
            "../envs/default.yaml"
        shell:
            """
            python {input.script:q} {input.slope:q} {input.shapes:q} {output:q} 2> {log:q}
            """

    rule clip_bathymetry:
        message:
            "Cut bathymetry data to the bounds of the input shapefile."
        input:
            script=workflow.source_path("../scripts/clip_raster.py"),
            shapes="resources/user/shapes/{shape}.parquet",
            bathymetry=rules.download_bathymetry.output,
        output:
            "resources/automatic/cutout/{shape}/bathymetry.tif",
        log:
            "logs/{shape}/clip_bathymetry.log",
        conda:
            "../envs/default.yaml"
        shell:
            """
            python {input.script:q} {input.bathymetry:q} {input.shapes:q} {output:q} 2> {log:q}
            """


##
# Globcover
##


rule download_globcover:
    message:
        "Download the GlobCover land cover data (~380 MB)."
    params:
        url=internal["resources"]["automatic"]["globcover"],
    output:
        "resources/automatic/global/globcover.zip",
    log:
        "logs/download_globcover.log",
    conda:
        "../envs/shell.yaml"
    shell:
        """
        curl -sSLo {output:q} {params.url:q}
        """


rule unzip_globcover:
    message:
        "Unzip the relevant TIF files from the GlobCover zip file."
    params:
        target_file=internal["resources"]["automatic"]["globcover_landcover_tif"],
    input:
        script=workflow.source_path("../scripts/unzip_like.py"),
        zipfile=rules.download_globcover.output,
    output:
        "resources/automatic/global/globcover-landcover.tif",
    log:
        "logs/unzip_globcover.log",
    conda:
        "../envs/shell.yaml"
    shell:
        """
        python {input.script:q} {input.zipfile:q} -f {params.target_file:q} -o {output:q} 2> {log:q}
        """


rule clip_landcover:
    message:
        "Cut land cover data to the bounds of the input shapefile."
    input:
        script=workflow.source_path("../scripts/clip_raster.py"),
        shapes="resources/user/shapes/{shape}.parquet",
        landcover=rules.unzip_globcover.output,
    output:
        "resources/automatic/cutout/{shape}/landcover.tif",
    log:
        "logs/{shape}/clip_landcover.log",
    conda:
        "../envs/default.yaml"
    shell:
        """
        python {input.script:q} {input.landcover:q} {input.shapes:q} {output:q} 2> {log:q}
        """


##
# Global Human Settlement Layer (GHSL)
##


rule download_ghsl:
    message:
        "Download the GHSL (Global Human Settlement Layer) built-up surface data."
    params:
        url=internal["resources"]["automatic"]["ghsl"],
    output:
        "resources/automatic/global/ghsl_built_s.zip",
    log:
        "logs/download_ghsl.log",
    conda:
        "../envs/shell.yaml"
    shell:
        """
        curl -sSLo {output:q} {params.url:q}
        """


rule unzip_ghsl:
    message:
        "Unzip the relevant TIF file from the GHSL data."
    params:
        target_file=internal["resources"]["automatic"]["ghsl_tif"],
    input:
        script=workflow.source_path("../scripts/unzip_like.py"),
        zipfile=rules.download_ghsl.output,
    output:
        "resources/automatic/global/ghsl_built_s.tif",
    log:
        "logs/unzip_ghsl.log",
    conda:
        "../envs/shell.yaml"
    shell:
        """
        python {input.script:q} {input.zipfile:q} -f {params.target_file:q} -o {output:q} 2> {log:q}
        """


rule clip_settlement:
    message:
        "Cut settlement data to the bounds of the input shapefile."
    input:
        script=workflow.source_path("../scripts/clip_raster.py"),
        shapes="resources/user/shapes/{shape}.parquet",
        settlement=rules.unzip_ghsl.output,
    output:
        "resources/automatic/cutout/{shape}/settlement.tif",
    log:
        "logs/{shape}/clip_settlement.log",
    conda:
        "../envs/default.yaml"
    shell:
        """
        python {input.script:q} {input.settlement:q} {input.shapes:q} {output:q} 2> {log:q}
        """


##
# Protected Areas (WDPA)
##


rule rasterise_clip_wdpa:
    message:
        "Rasterise and cut WDPA data to the bounds of the input shapefile, using the landcover raster as reference for the rasterisation."
    input:
        script=workflow.source_path("../scripts/clip_and_rasterise_polys.py"),
        shapes="resources/user/shapes/{shape}.parquet",
        reference_raster=rules.clip_landcover.output,
        protected_areas="resources/user/wdpa.gdb",
    output:
        "resources/automatic/cutout/{shape}/wdpa.tif",
    log:
        "logs/{shape}/clip_wdpa.log",
    conda:
        "../envs/default.yaml"
    shell:
        """
        python {input.script:q} {input.shapes:q} {input.reference_raster:q} {input.protected_areas:q} {output:q} 2> {log:q}
        """


##
# Solar Atlas
##

rule download_solar_atlas:
    message:
        "Download the Solar Atlas data."
    params:
        url=internal["resources"]["automatic"]["solar_atlas"],
    output:
        path="resources/automatic/global/solar_atlas.zip",
    log:
        "logs/download_solar_atlas.log",
    conda:
        "../envs/shell.yaml"
    shell:
        """
        curl -sSLo {output:q} {params.url:q}
        """

rule unzip_solar_atlas:
    message:
        "Unzip the relevant TIF files from the solar_atlas zip file."
    params:
        target_file=internal["resources"]["automatic"]["solar_atlas_tif"],
    input:
        script=workflow.source_path("../scripts/unzip_like.py"),
        zipfile=rules.download_solar_atlas.output,
    output:
        "resources/automatic/global/PVOUT.tif",
    log:
        "logs/unzip_solar_atlas.log",
    conda:
        "../envs/shell.yaml"
    shell:
        """
        python {input.script:q} {input.zipfile:q} -f {params.target_file:q} -o {output:q} 2> {log:q}
        """


rule clip_solar_atlas:
    message:
        "Cut solar atlas data to the bounds of the input shapefile."
    input:
        script=workflow.source_path("../scripts/clip_raster.py"),
        shapes="resources/user/shapes/{shape}.parquet",
        solar_atlas=rules.unzip_solar_atlas.output,
    output:
        "resources/automatic/cutout/{shape}/pv_out.tif",
    log:
        "logs/{shape}/clip_solar_atlas.log",
    conda:
        "../envs/default.yaml"
    shell:
        """
        python {input.script:q} {input.solar_atlas:q} {input.shapes:q} {output:q} 2> {log:q}
        """


##
# Wind Atlas
##

rule download_wind_atlas:
    message:
        "Download the Wind Atlas data."
    params:
        url=internal["resources"]["automatic"]["wind_atlas"],
    output:
        path="resources/automatic/global/WINDOUT.tif",
    log:
        "logs/download_wind_atlas.log",
    conda:
        "../envs/shell.yaml"
    shell:
        """
        wget --user-agent="Mozilla/5.0" \
            --tries=inf \
            --continue \
            {params.url:q} \
            -O {output:q}
        """


rule clip_wind_atlas:
    message:
        "Cut wind atlas data to the bounds of the input shapefile."
    input:
        script=workflow.source_path("../scripts/clip_raster.py"),
        shapes="resources/user/shapes/{shape}.parquet",
        wind_atlas=rules.download_wind_atlas.output,
    output:
        "resources/automatic/cutout/{shape}/wind_out.tif",
    log:
        "logs/{shape}/clip_wind_atlas.log",
    conda:
        "../envs/default.yaml"
    shell:
        """
        python {input.script:q} {input.wind_atlas:q} {input.shapes:q} {output:q} 2> {log:q}
        """
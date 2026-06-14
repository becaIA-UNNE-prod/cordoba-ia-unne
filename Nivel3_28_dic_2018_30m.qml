<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="AllStyleCategories" hasScaleBasedVisibilityFlag="0" minScale="1e+08" version="3.6.0-Noosa" maxScale="0">
  <flags>
    <Identifiable>1</Identifiable>
    <Removable>1</Removable>
    <Searchable>1</Searchable>
  </flags>
  <customproperties>
    <property value="false" key="WMSBackgroundLayer"/>
    <property value="false" key="WMSPublishDataSourceUrl"/>
    <property value="0" key="embeddedWidgets/count"/>
    <property value="Value" key="identify/format"/>
  </customproperties>
  <pipe>
    <rasterrenderer classificationMax="27" type="singlebandpseudocolor" alphaBand="-1" opacity="1" classificationMin="0" band="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Exact</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <rastershader>
        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0">
          <colorramp type="gradient" name="[source]">
            <prop k="color1" v="215,25,28,255"/>
            <prop k="color2" v="43,131,186,255"/>
            <prop k="discrete" v="0"/>
            <prop k="rampType" v="gradient"/>
            <prop k="stops" v="0.25;253,174,97,255:0.5;255,255,191,255:0.75;171,221,164,255"/>
          </colorramp>
          <item label="0" value="0" alpha="255" color="#ffffff"/>
          <item label="Monte" value="1" alpha="255" color="#009800"/>
          <item label="Arbustales y matorrales" value="2" alpha="255" color="#65d92b"/>
          <item label="Pastizal natural" value="3" alpha="255" color="#c94ed0"/>
          <item label="Pastizal natural con rocas o suelo desnudo" value="4" alpha="255" color="#f560ff"/>
          <item label="Rocas" value="5" alpha="255" color="#9d9d9d"/>
          <item label="Suelo desnudo" value="6" alpha="255" color="#ebebeb"/>
          <item label="Salina" value="7" alpha="255" color="#e2e2e2"/>
          <item label="Cuerpos de agua" value="8" alpha="255" color="#5970a1"/>
          <item label="Zonas anegables" value="9" alpha="255" color="#acdee9"/>
          <item label="Cursos de agua" value="10" alpha="255" color="#0000ff"/>
          <item label="Zona urbana consolidada" value="11" alpha="255" color="#cd1010"/>
          <item label="Zona urbana en proceso de consolidación" value="12" alpha="255" color="#e54949"/>
          <item label="Zona urbana sin consolidar" value="13" alpha="255" color="#ff9898"/>
          <item label="Infraestructura vial" value="14" alpha="255" color="#000000"/>
          <item label="Trigo" value="15" alpha="255" color="#e4ca00"/>
          <item label="Maíz" value="16" alpha="255" color="#e97f02"/>
          <item label="Soja" value="17" alpha="255" color="#a8ae00"/>
          <item label="Maní" value="18" alpha="255" color="#bd5a50"/>
          <item label="Sorgo" value="19" alpha="255" color="#76274f"/>
          <item label="Trigo-Maíz de segunda" value="20" alpha="255" color="#cdcd73"/>
          <item label="Trigo-Soja de segunda" value="21" alpha="255" color="#888c3c"/>
          <item label="Cultivos anuales irrigados" value="22" alpha="255" color="#fffd2e"/>
          <item label="Pasturas implantadas" value="23" alpha="255" color="#07eb98"/>
          <item label="Pasturas naturales manejadas" value="24" alpha="255" color="#b8ec98"/>
          <item label="Plantaciones forestales maderables" value="25" alpha="255" color="#84371d"/>
          <item label="Plantaciones perennes (frutales) de secano" value="26" alpha="255" color="#a8583d"/>
          <item label="Plantaciones perennes (frutales) irrigadas" value="27" alpha="255" color="#b07a67"/>
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="-1" contrast="0"/>
    <huesaturation colorizeRed="255" colorizeBlue="128" grayscaleMode="0" colorizeOn="0" colorizeStrength="100" saturation="0" colorizeGreen="128"/>
    <rasterresampler maxOversampling="2"/>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
